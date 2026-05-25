#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Causal VL dataset loader for PPO and SFT.

Supports annotation fields "path" or "paths" where each entry may be:
  1) a path to an image/video file
  2) a path to a folder (images only)
  3) a list of paths to images (all images; sorted)

Rules (as specified):
- Folder inputs: read ALL images (jpg/jpeg/png) in the folder, sorted as a sequence.
- Supported formats: images: jpg/jpeg/png; videos: mp4/mov/avi/mkv/webm.
- Video input: sample N frames uniformly (default 5), store extracted frames under the data/ directory
  in a subfolder named "<id>_frames", and pass those frames as the sequence.
- Max sequence length: 12. If longer, raise an error.
- Ordering: sort filenames alphanumerically for folders and lists.
- Mixed inputs (e.g., images + videos, folder containing non-image files, list containing mixed types): raise an error.
- If no valid media is found: raise an error.

On any rule violation, this loader raises RuntimeError to abort.
"""

import os
import glob
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image  # for simple existence checks and potential use by callers
import random


IMG_EXT = {".jpg", ".jpeg", ".png"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXT


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VID_EXT


def _list_images_in_folder(folder: str) -> List[str]:
    entries = sorted(os.listdir(folder))
    paths: List[str] = []
    for name in entries:
        p = os.path.join(folder, name)
        if os.path.isdir(p):
            # Mixed content (subfolders) not allowed
            raise RuntimeError(f"Folder contains subfolder (mixed content): {p}")
        if not _is_image(p):
            raise RuntimeError(f"Folder contains non-image file: {p}")
        paths.append(p)
    if not paths:
        raise RuntimeError(f"Folder has no images: {folder}")
    return paths


def _uniform_frame_indices(total: int, n: int) -> List[int]:
    if n <= 1:
        return [total // 2]
    return [min(total - 1, max(0, round(i * (total - 1) / (n - 1)))) for i in range(n)]


def _extract_video_frames(video_path: str, out_dir: str, num_frames: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    # If already present enough frames, reuse
    existing = sorted([
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith(base + "_f") and f.lower().endswith(".jpg")
    ])
    if len(existing) >= num_frames:
        return existing[:num_frames]

    frames: List[str] = []
    # Try OpenCV first
    cap = None
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        if total <= 0:
            raise RuntimeError("Unknown/empty frame count")
        indices = _uniform_frame_indices(total, num_frames)
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            # BGR -> RGB
            frame = frame[:, :, ::-1]
            out_path = os.path.join(out_dir, f"{base}_f{i:03d}.jpg")
            try:
                Image.fromarray(frame).save(out_path, format="JPEG", quality=90)
            except Exception:
                frame_bgr = frame[:, :, ::-1]
                cv2.imwrite(out_path, frame_bgr)
            frames.append(out_path)
    except Exception:
        # Fallback to imageio
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            import imageio.v3 as iio  # type: ignore
            from PIL import Image as _PILImage  # type: ignore
            meta = iio.immeta(video_path)
            raw_total = meta.get("nframes")
            try:
                total = int(raw_total) if raw_total is not None else 0
                if total == float("inf"):
                    total = 0
            except Exception:
                total = 0

            if total <= 0:
                # Unknown frame count: reservoir sample K frames uniformly during single pass
                reservoir = []  # list of (idx, frame)
                for idx, frame in enumerate(iio.imiter(video_path)):
                    if idx < num_frames:
                        reservoir.append((idx, frame))
                    else:
                        j = random.randint(0, idx)
                        if j < num_frames:
                            reservoir[j] = (idx, frame)
                if not reservoir:
                    raise RuntimeError("No frames read from video")
                # Sort by original index to preserve temporal order
                reservoir.sort(key=lambda x: x[0])
                for i, (_, frame) in enumerate(reservoir[:num_frames]):
                    out_path = os.path.join(out_dir, f"{base}_f{i:03d}.jpg")
                    _PILImage.fromarray(frame).save(out_path, format="JPEG", quality=90)
                    frames.append(out_path)
            else:
                indices = _uniform_frame_indices(total, num_frames)
                for i, idx in enumerate(indices):
                    frame = iio.imread(video_path, index=idx)
                    out_path = os.path.join(out_dir, f"{base}_f{i:03d}.jpg")
                    _PILImage.fromarray(frame).save(out_path, format="JPEG", quality=90)
                    frames.append(out_path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract frames from video {video_path}: {e}")
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frames[:num_frames]


@dataclass
class Sample:
    category: str
    subcategory: str
    ann_path: str
    gt_graph: Dict[str, Any]
    media_paths: List[str]  # resolved sequence of image paths


class CausalVLDataset:
    def __init__(self, root: str, num_video_frames: int = 5, max_images: int = 12):
        self.root = root
        self.num_video_frames = max(1, int(num_video_frames))
        self.max_images = int(max_images)
        self.samples: List[Sample] = []
        self._index()

    def _resolve_media(self, ann: Dict[str, Any], data_dir: str, ann_path: str = "") -> List[str]:
        # Get candidate paths from annotation
        raw = ann.get("paths")
        if raw is None:
            raw = ann.get("path")
        if raw is None:
            raise RuntimeError(f"Annotation has no 'path' or 'paths' in file: {ann_path}")

        # Normalize to list
        if isinstance(raw, str):
            raw_list = [raw]
        elif isinstance(raw, list):
            raw_list = list(raw)
        else:
            raise RuntimeError(f"'path/paths' must be string or list of strings in file: {ann_path}")

        # Resolve absolute paths
        abs_list: List[str] = []
        for p in raw_list:
            if not isinstance(p, str) or not p:
                raise RuntimeError(f"Invalid path entry in annotation file: {ann_path}")
            ap = p if os.path.isabs(p) else os.path.join(data_dir, p)
            if not os.path.exists(ap):
                raise RuntimeError(f"Media path does not exist: {ap} (from file: {ann_path})")
            abs_list.append(ap)

        # Determine type consistency
        types = []
        for ap in abs_list:
            if os.path.isdir(ap):
                types.append("folder")
            elif _is_image(ap):
                types.append("image")
            elif _is_video(ap):
                types.append("video")
            else:
                raise RuntimeError(f"Unsupported media type: {ap}")

        if len(set(types)) != 1:
            raise RuntimeError(f"Mixed input types in paths: {types}")

        media_paths: List[str] = []
        kind = types[0]

        if kind == "folder":
            if len(abs_list) != 1:
                raise RuntimeError("Multiple folders provided; only one allowed")
            folder = abs_list[0]
            imgs = _list_images_in_folder(folder)
            if len(imgs) > self.max_images:
                raise RuntimeError(f"Folder yields >{self.max_images} images: {folder}")
            media_paths = imgs

        elif kind == "image":
            # Ensure all images; sort
            imgs = sorted(abs_list)
            if len(imgs) == 0:
                raise RuntimeError("No images found")
            if len(imgs) > self.max_images:
                raise RuntimeError(f"List yields >{self.max_images} images")
            media_paths = imgs

        elif kind == "video":
            if len(abs_list) != 1:
                raise RuntimeError("Multiple videos provided; only one allowed")
            video = abs_list[0]
            # Determine frames output dir under data: <data_dir>/<id>_frames
            sid = str(ann.get("id") or os.path.splitext(os.path.basename(video))[0])
            out_dir = os.path.join(data_dir, f"{sid}_frames")
            
            # Check if pre-sampled frames already exist
            if os.path.exists(out_dir):
                existing_frames = _list_images_in_folder(out_dir)
                if existing_frames:
                    # Use existing pre-sampled frames
                    frames = existing_frames
                    # print(f"[Info] Using pre-sampled frames from {out_dir} ({len(frames)} frames)")
                else:
                    # Directory exists but is empty, extract frames
                    frames = _extract_video_frames(video, out_dir, self.num_video_frames)
                    print(f"[Info] Extracted {len(frames)} frames to {out_dir}")
            else:
                # No pre-sampled frames, extract them
                frames = _extract_video_frames(video, out_dir, self.num_video_frames)
                print(f"[Info] Extracted {len(frames)} frames to {out_dir}")
            
            if len(frames) > self.max_images:
                raise RuntimeError(f"Video frames >{self.max_images} for {video}")
            media_paths = frames

        else:
            raise RuntimeError(f"Unknown media kind: {kind}")

        if not media_paths:
            raise RuntimeError("Resolved media sequence is empty")
        return media_paths

    def _index(self) -> None:
        pattern = os.path.join(self.root, "*", "*", "annotations", "*.json")
        for ann_path in glob.glob(pattern):
            sub_dir = os.path.dirname(os.path.dirname(ann_path))  # .../<subcategory>
            cat_dir = os.path.dirname(sub_dir)                    # .../<category>
            category = os.path.basename(cat_dir)
            subcategory = os.path.basename(sub_dir)
            data_dir = os.path.join(self.root, category, subcategory, "data")

            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)

                media_paths = self._resolve_media(ann, data_dir, ann_path)
                self.samples.append(Sample(
                    category=category,
                    subcategory=subcategory,
                    ann_path=ann_path,
                    gt_graph=ann,
                    media_paths=media_paths,
                ))
            except Exception as e:
                # Log the reason and the annotation path, then skip
                print(f"[Dataset] Skipping annotation due to error: {e}. File: {ann_path}")
                continue


def rationale_path_for_annotation(ann_path: str) -> str:
    """Return the expected rationale text path for an annotation JSON.

    <root>/<cat>/<sub>/annotations/<file>.json -> <root>/<cat>/<sub>/rationales/<file>.txt
    """
    sub_dir = os.path.dirname(os.path.dirname(ann_path))          # .../<cat>/<sub>
    cat_dir = os.path.dirname(sub_dir)                            # .../<cat>
    root = os.path.dirname(cat_dir)
    category = os.path.basename(cat_dir)
    subcategory = os.path.basename(sub_dir)
    base = os.path.splitext(os.path.basename(ann_path))[0] + ".txt"
    return os.path.join(root, category, subcategory, "rationales", base)


