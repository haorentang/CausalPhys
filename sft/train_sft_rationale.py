#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT on golden rationales for Qwen2-VL with weighted loss.

Supervises generating the rationale paragraph followed by the final answer letter.
Dataset/loader mirrors PPO dataset structure. For each sample we build a chat:
  - user: images + question + explicit instruction to output a rationale then a single choice letter
  - assistant: rationale paragraph + newline + choice letter

We evaluate per epoch on a held-out split by accuracy of predicted choice.

Weighted Loss Feature:
- Rationale tokens get lower weight (default: 1.0)
- Answer tokens get higher weight (default: 2.0)
- This encourages the model to focus more on getting the final answer correct
- Use --rationale_weight and --answer_weight to customize the weights
"""

import os
import gc
import math
import random
import argparse
import json
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers.trainer_utils import get_last_checkpoint

from dataset.causal_vl import CausalVLDataset

# Import shared evaluation utilities
from .eval_utils import evaluate_accuracy,ValEvalCallback, _read_text, build_model_and_processor, split_items_by_subcategory


class WeightedSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that supports weighted loss for rationale vs answer tokens.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store weights for debugging
        self.rationale_weight = getattr(self.data_collator, 'rationale_weight', 1.0)
        self.answer_weight = getattr(self.data_collator, 'answer_weight', 2.0)
        # Enable training accuracy logging
        self.log_train_accuracy = True
        self._train_accuracy_sum = 0.0
        self._train_accuracy_count = 0
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss where rationale tokens have lower weight and answer tokens have higher weight.
        """
        if "loss_weights" in inputs:
            # Get the standard loss computation
            labels = inputs.get("labels")
            loss_weights = inputs.get("loss_weights")
            
            # Check for empty rationale or answer sections and warn
            sample_labels = labels[0]
            sample_weights = loss_weights[0]
            supervised_mask = (sample_labels != -100)
            
            # Check rationale section
            rationale_mask = (sample_weights == self.rationale_weight) & supervised_mask
            rationale_count = rationale_mask.sum().item()
            
            # Check answer section
            answer_mask = (sample_weights == self.answer_weight) & supervised_mask
            answer_count = answer_mask.sum().item()
            
            # Warn if sections are empty
            if rationale_count == 0:
                print("⚠️  WARNING: No rationale tokens found in supervised region!")
            if answer_count == 0:
                print("⚠️  WARNING: No answer tokens found in supervised region!")
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Compute standard cross entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_loss_weights = loss_weights[..., 1:].contiguous()
            
            # Flatten the tensors
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_loss_weights = shift_loss_weights.view(-1)
            

            # Compute loss only for non-ignored tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits, shift_labels)
            
            # Apply weights and compute mean
            weighted_losses = losses * shift_loss_weights
            
            
            # Only average over non-zero weights (non-padding tokens)
            non_zero_weights = (shift_loss_weights > 0).float()
            if non_zero_weights.sum() > 0:
                loss = weighted_losses.sum() / non_zero_weights.sum()
            else:
                loss = torch.tensor(0.0, device=losses.device, requires_grad=True)
            
            # Store training loss for epoch logging
            if hasattr(self, '_train_loss_count'):
                self._train_loss_sum += loss.item()
                self._train_loss_count += 1
            # Compute training accuracy from model outputs
            if hasattr(self, 'log_train_accuracy') and self.log_train_accuracy:
                with torch.no_grad():
                    # Get predictions for supervised tokens only
                    supervised_mask = (shift_labels != -100)
                    if supervised_mask.sum() > 0:
                        supervised_logits = shift_logits[supervised_mask]
                        supervised_labels = shift_labels[supervised_mask]
                        supervised_weights = shift_loss_weights[supervised_mask]
                        
                        # Debug: Check dimensions (only for first few batches)
                        if hasattr(self, '_debug_train_count'):
                            self._debug_train_count += 1
                        else:
                            self._debug_train_count = 1
                        
                        if self._debug_train_count <= 3:
                            print(f"[DEBUG] Train batch {self._debug_train_count}:")
                            print(f"  shift_logits shape: {shift_logits.shape}")
                            print(f"  shift_labels shape: {shift_labels.shape}")
                            print(f"  supervised_mask shape: {supervised_mask.shape}")
                            print(f"  supervised_logits shape: {supervised_logits.shape}")
                            print(f"  supervised_labels shape: {supervised_labels.shape}")
                        
                        predictions = torch.argmax(supervised_logits, dim=-1)
                        
                        # Compute overall accuracy
                        correct = (predictions == supervised_labels).float().sum()
                        total = supervised_mask.sum().float()
                        train_accuracy = correct / total
                        
                        # Compute rationale accuracy (tokens with rationale_weight)
                        rationale_mask = (supervised_weights == self.rationale_weight)
                        if rationale_mask.sum() > 0:
                            rationale_correct = (predictions[rationale_mask] == supervised_labels[rationale_mask]).float().sum()
                            rationale_total = rationale_mask.sum().float()
                            rationale_accuracy = rationale_correct / rationale_total
                        else:
                            rationale_accuracy = torch.tensor(0.0)
                        
                        # Compute result accuracy (tokens with answer_weight)
                        result_mask = (supervised_weights == self.answer_weight)
                        if result_mask.sum() > 0:
                            result_correct = (predictions[result_mask] == supervised_labels[result_mask]).float().sum()
                            result_total = result_mask.sum().float()
                            result_accuracy = result_correct / result_total
                        else:
                            result_accuracy = torch.tensor(0.0)
                        
                        # Store for later logging
                        if not hasattr(self, '_train_accuracy_sum'):
                            self._train_accuracy_sum = 0.0
                            self._train_accuracy_count = 0
                        if not hasattr(self, '_train_rationale_accuracy_sum'):
                            self._train_rationale_accuracy_sum = 0.0
                            self._train_rationale_accuracy_count = 0
                        if not hasattr(self, '_train_result_accuracy_sum'):
                            self._train_result_accuracy_sum = 0.0
                            self._train_result_accuracy_count = 0
                            
                        self._train_accuracy_sum += train_accuracy.item()
                        self._train_accuracy_count += 1
                        self._train_rationale_accuracy_sum += rationale_accuracy.item()
                        self._train_rationale_accuracy_count += 1
                        self._train_result_accuracy_sum += result_accuracy.item()
                        self._train_result_accuracy_count += 1
            
            return (loss, outputs) if return_outputs else loss
        else:
            # Fallback to standard loss computation
            raise ValueError("loss_weights not found in inputs")
    
    def log_accumulated_train_accuracy(self):
        """Log the accumulated training accuracy and reset counters."""
        if hasattr(self, '_train_accuracy_count') and self._train_accuracy_count > 0:
            avg_accuracy = self._train_accuracy_sum / self._train_accuracy_count
            print(f"[Train] Average training accuracy: {avg_accuracy * 100:.2f}%")
            
            # Log rationale and result accuracy separately
            if hasattr(self, '_train_rationale_accuracy_count') and self._train_rationale_accuracy_count > 0:
                avg_rationale_accuracy = self._train_rationale_accuracy_sum / self._train_rationale_accuracy_count
                print(f"[Train] Average rationale accuracy: {avg_rationale_accuracy * 100:.2f}%")
            
            if hasattr(self, '_train_result_accuracy_count') and self._train_result_accuracy_count > 0:
                avg_result_accuracy = self._train_result_accuracy_sum / self._train_result_accuracy_count
                print(f"[Train] Average result accuracy: {avg_result_accuracy * 100:.2f}%")
            
            # Reset counters
            self._train_accuracy_sum = 0.0
            self._train_accuracy_count = 0
            if hasattr(self, '_train_rationale_accuracy_sum'):
                self._train_rationale_accuracy_sum = 0.0
                self._train_rationale_accuracy_count = 0
            if hasattr(self, '_train_result_accuracy_sum'):
                self._train_result_accuracy_sum = 0.0
                self._train_result_accuracy_count = 0
            if hasattr(self, '_train_loss_sum'):
                self._train_loss_sum = 0.0
                self._train_loss_count = 0
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute eval_loss only."""
        # Call parent evaluate to get standard metrics
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # For rationale training, we rely on the ValEvalCallback for proper task-level evaluation
        # The ValEvalCallback does generation and checks if the final answer letter is correct
        # This is much more appropriate than token-level accuracy for rationale generation
        
        return metrics
    


class TrainAccuracyCallback(TrainerCallback):
    """Callback to log training accuracy at the end of each epoch."""
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log training accuracy at the end of each epoch."""
        trainer = kwargs.get('trainer')
        if hasattr(trainer, 'log_accumulated_train_accuracy'):
            trainer.log_accumulated_train_accuracy()


class EpochMetricsLogger(TrainerCallback):
    """Callback to log epoch metrics to a file in the output directory."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "epoch_metrics.log")
        self._write_header()
    
    def _write_header(self):
        """Write header to the log file."""
        with open(self.log_file, "w") as f:
            f.write("Epoch,Train_Loss,Train_Accuracy,Val_Loss,Val_Accuracy,Timestamp\n")
    
    def _log_metrics(self, epoch: int, train_loss: float = None, train_acc: float = None, 
                    val_loss: float = None, val_acc: float = None):
        """Log metrics to file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_loss_str = f"{train_loss:.6f}" if train_loss is not None else "N/A"
        train_acc_str = f"{train_acc:.4f}" if train_acc is not None else "N/A"
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        
        with open(self.log_file, "a") as f:
            f.write(f"{epoch},{train_loss_str},{train_acc_str},{val_loss_str},{val_acc_str},{timestamp}\n")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log metrics at the end of each epoch."""
        trainer = kwargs.get('trainer')
        epoch = int(state.epoch)
        
        # Get training metrics
        train_loss = None
        train_acc = None
        if hasattr(trainer, '_train_loss_count') and trainer._train_loss_count > 0:
            train_loss = trainer._train_loss_sum / trainer._train_loss_count
        if hasattr(trainer, '_train_accuracy_count') and trainer._train_accuracy_count > 0:
            train_acc = trainer._train_accuracy_sum / trainer._train_accuracy_count
        
        # Get validation metrics from trainer state
        val_loss = None
        val_acc = None
        if hasattr(trainer, 'state') and trainer.state.log_history:
            # Look for the most recent eval metrics
            for log_entry in reversed(trainer.state.log_history):
                if 'eval_loss' in log_entry:
                    val_loss = log_entry['eval_loss']
                    break
        
        # Log to file
        self._log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Also print to console
        print(f"\n[Epoch {epoch}] Metrics logged to {self.log_file}")
        if train_loss is not None:
            print(f"  Train Loss: {train_loss:.6f}")
        if train_acc is not None:
            print(f"  Train Accuracy: {train_acc:.4f}")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.6f}")
        if val_acc is not None:
            print(f"  Val Accuracy: {val_acc:.4f}")


CHOICE_LETTERS = ["A", "B", "C", "D"]


# Import shared types
from .shared_types import SFTItem


def _open_image(p: str, resize_to: Optional[tuple] = None) -> Optional[Image.Image]:
    try:
        img = Image.open(p).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


def rationale_path_for_annotation(ann_path: str, rationale_subdir: str = "rationales") -> str:
    """Convert annotation path to rationale file path."""
    # Convert: dataset/Perception/stability/annotations/001035.json
    # To: dataset/Perception/stability/<rationale_subdir>/001035.txt
    ann_path = ann_path.replace("/annotations/", f"/{rationale_subdir}/")
    ann_path = ann_path.replace(".json", ".txt")
    return ann_path


def _parse_rationale_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception:
        return ""
    # Remove newlines and strip whitespace
    return txt.replace('\n', '').strip()


def build_answer_only_items(dataset_root: str, num_video_frames: int, max_images: int = 8, resize_to: Optional[tuple] = None) -> List[SFTItem]:
    """
    Build items for answer-only training (first stage of two-stage training).
    This creates items without rationales, only with questions and answers.
    """
    ds = CausalVLDataset(dataset_root, num_video_frames=num_video_frames)
    items: List[SFTItem] = []
    skipped_count = 0
    item_index = 0  # Global index counter for all valid items
    for i, s in enumerate(ds.samples):
        q = s.gt_graph.get("question")
        gt = s.gt_graph.get("ground_truth_answer")
        if not isinstance(q, str) or not isinstance(gt, str):
            skipped_count += 1
            if skipped_count <= 5:  # Only print first 5 skipped items
                print(f"⚠️  Skipping sample {i}: question={type(q)}, gt={type(gt)} | ann: {s.ann_path}")
            continue
        gt = gt.strip().upper()
        if gt not in CHOICE_LETTERS:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"⚠️  Skipping sample {i}: invalid answer letter '{gt}' | ann: {s.ann_path}")
            continue
        imgs: List[Image.Image] = []
        for p in (s.media_paths or [])[:max_images]:
            img = _open_image(p, resize_to=resize_to)
            if img is not None:
                imgs.append(img)
        if not imgs:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"⚠️  Skipping sample {i}: no valid images | ann: {s.ann_path}")
            continue
        # Create SFTItem with empty rationale for answer-only training
        items.append(SFTItem(images=imgs, question=q, answer_letter=gt, rationale="", category=s.category, subcategory=s.subcategory, index=item_index, ann_path=s.ann_path))
        item_index += 1
    
    print(f"[Info] Built {len(items)} answer-only items, skipped {skipped_count} invalid samples")
    return items


def build_items(dataset_root: str, num_video_frames: int, max_images: int = 8, resize_to: Optional[tuple] = None, rationale_subdir: str = "rationales") -> List[SFTItem]:
    ds = CausalVLDataset(dataset_root, num_video_frames=num_video_frames)
    items: List[SFTItem] = []
    skipped_count = 0
    item_index = 0  # Global index counter for all valid items
    for i, s in enumerate(ds.samples):
        q = s.gt_graph.get("question")
        gt = s.gt_graph.get("ground_truth_answer")
        if not isinstance(q, str) or not isinstance(gt, str):
            skipped_count += 1
            if skipped_count <= 5:  # Only print first 5 skipped items
                print(f"⚠️  Skipping sample {i}: question={type(q)}, gt={type(gt)} | ann: {s.ann_path}")
            continue
        gt = gt.strip().upper()
        if gt not in CHOICE_LETTERS:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"⚠️  Skipping sample {i}: invalid answer letter '{gt}' | ann: {s.ann_path}")
            continue
        rat_path = rationale_path_for_annotation(s.ann_path, rationale_subdir=rationale_subdir)
        if not os.path.exists(rat_path):
            skipped_count += 1
            if skipped_count <= 5:
                print(f"⚠️  Skipping sample {i}: no rationale file | ann: {s.ann_path}")
            continue
        rationale = _parse_rationale_file(rat_path)
        imgs: List[Image.Image] = []
        for p in (s.media_paths or [])[:max_images]:
            img = _open_image(p, resize_to=resize_to)
            if img is not None:
                imgs.append(img)
        if not imgs:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"⚠️  Skipping sample {i}: no valid images | ann: {s.ann_path}")
            continue
        items.append(SFTItem(images=imgs, question=q, answer_letter=gt, rationale=rationale, category=s.category, subcategory=s.subcategory, index=item_index, ann_path=s.ann_path))
        item_index += 1
    
    print(f"[Info] Built {len(items)} items, skipped {skipped_count} invalid samples")
    return items


class TorchSFTDataset(Dataset):
    def __init__(self, items: List[SFTItem], is_answer_only: bool = False):
        self.items = items
        self.is_answer_only = is_answer_only
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        # Return a dict so TRL's SFTTrainer can introspect sample keys safely
        # Add validation to catch corrupted items
        if it.question is None or it.answer_letter is None or not it.images:
            raise ValueError(f"⚠️  Corrupted item at index {idx}: question={it.question is not None}, answer_letter={it.answer_letter is not None}, images={len(it.images) if it.images else 0}")
        
        # For answer-only training, rationale can be empty
        # For rationale training, rationale should not be empty
        if not self.is_answer_only and (it.rationale is None or not it.rationale.strip()):
            raise ValueError(f"⚠️  Corrupted item at index {idx}: rationale is empty or None (expected for rationale training)")
        
        # Return structured fields for cleaner processing
        result = {
            "images": it.images,
            "question": it.question,
            "answer_letter": it.answer_letter,
            "rationale": it.rationale,
        }
        return result


class QwenVLAnswerOnlyCollator:
    """
    Collator for answer-only training (first stage of two-stage training).
    This is a simplified version that only supervises the final answer letter.
    """
    def __init__(self, processor, tokenizer, device: torch.device, system_text: str, user_template: str, debug_tokenization: bool = False):
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.system_text = system_text
        self.user_template = user_template
        self.debug_tokenization = debug_tokenization
        self._debug_count = 0

        # Debug: Print initialization
        if self.debug_tokenization:
            print(f"\n[DEBUG] QwenVLAnswerOnlyCollator initialized with debug_tokenization={debug_tokenization}")
            print(f"[DEBUG] System text length: {len(system_text)} chars")
            print(f"[DEBUG] User template: {user_template[:100]}...")

    def __call__(self, batch: List[dict]):
        full_texts, user_texts, images_per = [], [], []
        
        for it in batch:
            # Extract structured fields from batch item
            images = it.get("images") if isinstance(it, dict) else it.images
            question = it.get("question") if isinstance(it, dict) else it.question
            answer_letter = it.get("answer_letter") if isinstance(it, dict) else it.answer_letter

            # Create the proper chat format for answer-only SFT training
            header_lines = [f"Question:\n{str(question).strip()}"]
            if images:
                header_lines.append(f"[Images provided: {len(images)}; consume in the given order as a sequence]")
            header = "\n\n".join(header_lines)
            user_content = [{"type": "image"} for _ in images] + [
                {"type": "text", "text": header}
            ]
            messages = [
                {"role": "system", "content": self.system_text},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"<result>{answer_letter}</result>"}
                ]},
            ]
            full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Skip invalid items
            if question is None or answer_letter is None or not images:
                if self.debug_tokenization:
                    print(f"⚠️  Skipping invalid item: question={question is not None}, answer_letter={answer_letter is not None}, images={len(images) if images else 0}")
                continue

            # Use the full_text for SFT training
            full_texts.append(full_text)
            
            # Create user text for masking (everything up to assistant response)
            user_messages = [
                {"role": "system", "content": self.system_text},
                {"role": "user", "content": user_content},
            ]
            user_text = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
            user_texts.append(user_text)
            # images list: processor accepts lists of images matching text entries
            images_per.append(images)

        # Check if all items were invalid
        if not full_texts:
            if self.debug_tokenization:
                print("⚠️  All items in batch were invalid, returning empty batch")
            # Return a minimal valid batch to avoid training errors
            return {
                "input_ids": torch.tensor([[0]], dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor([[1]], dtype=torch.long, device=self.device),
                "pixel_values": torch.zeros((1, 3, 448, 448), dtype=torch.float32, device=self.device),
                "labels": torch.tensor([[-100]], dtype=torch.long, device=self.device),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long, device=self.device),
            }

        # Flatten images: Qwen2-VL processor supports parallel multiple images per sample by giving list per sample
        enc = self.processor(
            text=full_texts,
            images=images_per,
            return_tensors="pt",
            padding=True
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        pixel_values = enc["pixel_values"]
        grid_thw = enc.get("image_grid_thw", None)

        # Supervise ALL generated tokens (everything after prompt_len)
        # For answer-only training, we only supervise the final answer
        labels = input_ids.clone()
        labels.fill_(-100)
        
        for i, (ut, imgs) in enumerate(zip(user_texts, images_per)):
            enc_prompt = self.processor(text=[ut], images=[imgs], return_tensors="pt", padding=False)
            prompt_len = enc_prompt["input_ids"].shape[1]
            # Effective sequence length ignoring padding
            seq_len = int(attention_mask[i].sum().item())

            if prompt_len < seq_len:
                # Initialize: mask all generated tokens first
                labels[i, prompt_len:seq_len] = -100
                
                # Create answer-only supervision mask
                generated_tokens = input_ids[i, prompt_len:seq_len]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                
                # Find result section
                result_start = generated_text.find('<result>')
                result_end = generated_text.find('</result>')
                
                # Debug: Print what we found
                if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                    print(f"\n--- ANSWER-ONLY TAG PARSING DEBUG ---")
                    print(f"Generated text: {generated_text[:200]}...")
                    print(f"result_start: {result_start}")
                    print(f"result_end: {result_end}")
                    print(f"--- END ANSWER-ONLY TAG PARSING DEBUG ---\n")
                
                if result_start != -1 and result_end != -1:
                    # Calculate token positions for result section (including tags)
                    result_section_start = result_start
                    result_section_end = result_end + len('</result>')
                    result_section_start_pos = prompt_len + len(self.tokenizer.encode(generated_text[:result_section_start], add_special_tokens=False))
                    result_section_end_pos = prompt_len + len(self.tokenizer.encode(generated_text[:result_section_end], add_special_tokens=False))
               
                    if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                        print(f"Answer-only section parsing details:")
                        print(f"  result_section_start: {result_section_start}, result_section_end: {result_section_end}")
                        print(f"  result_section_start_pos: {result_section_start_pos}, result_section_end_pos: {result_section_end_pos}")
                        print(f"  result section length: {result_section_end_pos - result_section_start_pos} tokens")
                    
                    # Supervise result section (including tags)
                    if result_section_start_pos < seq_len and result_section_end_pos <= seq_len:
                        result_section_end_pos = min(result_section_end_pos, seq_len)
                        labels[i, result_section_start_pos:result_section_end_pos] = input_ids[i, result_section_start_pos:result_section_end_pos]
                        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                            print(f"Supervised result section at positions {result_section_start_pos}:{result_section_end_pos}")
                    else:
                        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                            print(f"Result section out of bounds: start={result_section_start_pos}, end={result_section_end_pos}, seq_len={seq_len}")
                        raise ValueError("result section not found")
                else:
                    # Fallback: if we can't parse the structure, supervise all generated tokens
                    if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                        print(f"\n--- ANSWER-ONLY PARSING FAILED DEBUG ---")
                        print(f"Generated text: {generated_text[:200]}...")
                        print(f"Could not find result tags, supervising all generated tokens")
                        print(f"--- END ANSWER-ONLY PARSING FAILED DEBUG ---\n")
                    labels[i, prompt_len:seq_len] = input_ids[i, prompt_len:seq_len]

        # Ensure padding is ignored in loss (safeguard)
        labels = labels.masked_fill(attention_mask == 0, -100)

        # Debug: Check label masking effectiveness
        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:
            print(f"\n--- ANSWER-ONLY LABEL MASKING DEBUG ---")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Check first sample
            sample_input_ids = input_ids[0]
            sample_labels = labels[0]
            
            # Count masked vs supervised tokens
            masked_tokens = (sample_labels == -100).sum().item()
            supervised_tokens = (sample_labels != -100).sum().item()
            total_tokens = len(sample_input_ids)
            
            print(f"Total tokens: {total_tokens}")
            print(f"Masked tokens (user input): {masked_tokens}")
            print(f"Supervised tokens (answer only): {supervised_tokens}")
            print(f"Mask ratio: {masked_tokens/total_tokens:.2%}")
            
            # Show the supervised token IDs and their text
            supervised_indices = (sample_labels != -100).nonzero(as_tuple=True)[0]
            if len(supervised_indices) > 0:
                supervised_token_ids = sample_input_ids[supervised_indices]
                supervised_token_text = [self.tokenizer.decode([tid]) for tid in supervised_token_ids]
                print(f"Supervised token IDs: {supervised_token_ids.tolist()}")
                print(f"Supervised token text: {supervised_token_text}")
            
            print(f"--- END ANSWER-ONLY LABEL MASKING DEBUG ---\n")
        
        # Debug: Print tokenization details for first batch item
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:  # Only debug first 3 batches
            print(f"\n=== ANSWER-ONLY DEBUG BATCH {self._debug_count} ===")
            print(f"Full text: {full_texts[0][:200]}...")
            print(f"User text: {user_texts[0][:200]}...")
            
            # Get answer from the batch item
            answer_letter = batch[0].get("answer_letter") if isinstance(batch[0], dict) else batch[0].answer_letter
            print(f"Answer letter: '{answer_letter}'")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Show which tokens are supervised (not -100)
            supervised_mask = labels[0] != -100
            supervised_tokens = input_ids[0][supervised_mask]
            
            print(f"Supervised tokens: {supervised_tokens.tolist()}")
            print(f"Supervised token text: {[self.tokenizer.decode([tid]) for tid in supervised_tokens]}")
            print(f"Number of supervised tokens: {len(supervised_tokens)}")
            print(f"Expected: answer tokens only")
            
            print("=" * 50)

        # Keep tensors on CPU for Trainer to move; ensure standard dtypes
        pixel_values = pixel_values.to(dtype=torch.float32)
        if grid_thw is not None:
            if not isinstance(grid_thw, torch.Tensor):
                grid_thw = torch.tensor(grid_thw, dtype=torch.long)
            else:
                grid_thw = grid_thw.to(dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "image_grid_thw": grid_thw,
        }


class QwenVLRationaleCollator:
    def __init__(self, processor, tokenizer, device: torch.device, system_text: str, user_template: str, debug_tokenization: bool = False, 
                 rationale_weight: float = 1.0, answer_weight: float = 2.0):
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.system_text = system_text
        self.user_template = user_template
        self.debug_tokenization = debug_tokenization
        self.rationale_weight = rationale_weight
        self.answer_weight = answer_weight
        self._debug_count = 0

        # Debug: Print initialization
        if self.debug_tokenization:
            print(f"\n[DEBUG] QwenVLRationaleCollator initialized with debug_tokenization={debug_tokenization}")
            print(f"[DEBUG] System text length: {len(system_text)} chars")
            print(f"[DEBUG] User template: {user_template[:100]}...")
            print(f"[DEBUG] Loss weights - Rationale: {rationale_weight}, Answer: {answer_weight}")

    def __call__(self, batch: List[dict]):
        full_texts, user_texts, images_per = [], [], []
        
        for it in batch:
            # Extract structured fields from batch item
            images = it.get("images") if isinstance(it, dict) else it.images
            question = it.get("question") if isinstance(it, dict) else it.question
            answer_letter = it.get("answer_letter") if isinstance(it, dict) else it.answer_letter
            rationale = it.get("rationale") if isinstance(it, dict) else it.rationale

            # Create the proper chat format for SFT training
            header_lines = [f"Question:\n{str(question).strip()}"]
            if images:
                header_lines.append(f"[Images provided: {len(images)}; consume in the given order as a sequence]")
            header = "\n\n".join(header_lines)
            user_content = [{"type": "image"} for _ in images] + [
                {"type": "text", "text": header}
            ]
            messages = [
                {"role": "system", "content": self.system_text},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"<rationale>{rationale}</rationale>\n<result>{answer_letter}</result>"}
                ]},
            ]
            full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Debug: Print what we extracted from the batch item
            if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:
                print(f"🔍 Extracted from batch item: question={question is not None}, answer_letter={answer_letter is not None}, rationale={rationale is not None}, images={len(images) if images else 0}")
                print(f"🔍 Batch item keys: {list(it.keys()) if isinstance(it, dict) else 'not a dict'}")
                print(f"🔍 Batch item type: {type(it)}")
            
            # Debug: Check if dataset items are corrupted
            if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:
                print(f"\n--- DATASET ITEM DEBUG ---")
                print(f"Question: {question[:100] if question else 'None'}")
                print(f"Answer letter: '{answer_letter}'")
                print(f"Rationale: {rationale[:100] if rationale else 'None'}")
                print(f"Images count: {len(images) if images else 0}")
                print(f"Item type: {type(it)}")
                if isinstance(it, dict):
                    print(f"Item keys: {list(it.keys())}")
                print(f"--- END DATASET ITEM DEBUG ---\n")
            
            # Skip invalid items
            if question is None or answer_letter is None or not images or not rationale:
                if self.debug_tokenization:
                    print(f"⚠️  Skipping invalid item: question={question is not None}, answer_letter={answer_letter is not None}, images={len(images) if images else 0}, rationale={rationale is not None}")
                continue

            # Use the full_text for SFT training
            full_texts.append(full_text)
            
            # Create user text for masking (everything up to assistant response)
            user_messages = [
                {"role": "system", "content": self.system_text},
                {"role": "user", "content": user_content},
            ]
            user_text = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
            user_texts.append(user_text)
            # images list: processor accepts lists of images matching text entries
            images_per.append(images)

        # Check if all items were invalid
        if not full_texts:
            if self.debug_tokenization:
                print("⚠️  All items in batch were invalid, returning empty batch")
            # Return a minimal valid batch to avoid training errors
            return {
                "input_ids": torch.tensor([[0]], dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor([[1]], dtype=torch.long, device=self.device),
                "pixel_values": torch.zeros((1, 3, 448, 448), dtype=torch.float32, device=self.device),
                "labels": torch.tensor([[-100]], dtype=torch.long, device=self.device),
                "loss_weights": torch.tensor([[0.0]], dtype=torch.float32, device=self.device),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long, device=self.device),
            }

        # Flatten images: Qwen2-VL processor supports parallel multiple images per sample by giving list per sample
        enc = self.processor(
            text=full_texts,
            images=images_per,
            return_tensors="pt",
            padding=True
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        pixel_values = enc["pixel_values"]
        grid_thw = enc.get("image_grid_thw", None)

        # Supervise ALL generated tokens (everything after prompt_len)
        # This is more robust than trying to match specific rationale sequences
        # We want the model to generate the correct rationale AND the correct final answer
        labels = input_ids.clone()
        labels.fill_(-100)
        
        # Create loss weights tensor (same shape as labels)
        loss_weights = torch.ones_like(labels, dtype=torch.float32)
        
        for i, (ut, imgs) in enumerate(zip(user_texts, images_per)):
            enc_prompt = self.processor(text=[ut], images=[imgs], return_tensors="pt", padding=False)
            prompt_len = enc_prompt["input_ids"].shape[1]
            # Effective sequence length ignoring padding
            seq_len = int(attention_mask[i].sum().item())

            if prompt_len < seq_len:
                # Initialize: mask all generated tokens first
                labels[i, prompt_len:seq_len] = -100
                
                # Create weighted loss mask
                generated_tokens = input_ids[i, prompt_len:seq_len]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                
                # Find rationale and answer sections
                rationale_start = generated_text.find('<rationale>')
                rationale_end = generated_text.find('</rationale>')
                result_start = generated_text.find('<result>')
                result_end = generated_text.find('</result>')
                
                # Debug: Print what we found
                if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                    print(f"\n--- TAG PARSING DEBUG ---")
                    print(f"Generated text: {generated_text[:200]}...")
                    print(f"rationale_start: {rationale_start}")
                    print(f"rationale_end: {rationale_end}")
                    print(f"result_start: {result_start}")
                    print(f"result_end: {result_end}")
                    print(f"rationale_weight: {self.rationale_weight}")
                    print(f"answer_weight: {self.answer_weight}")
                    print(f"--- END TAG PARSING DEBUG ---\n")
                
                if rationale_start != -1 and rationale_end != -1 and result_start != -1 and result_end != -1:
                    # Calculate token positions for ENTIRE SECTIONS (including tags)
                    # Rationale section: from <rationale> to </rationale> (inclusive)
                    rationale_section_start = rationale_start
                    rationale_section_end = rationale_end + len('</rationale>')
                    rationale_section_start_pos = prompt_len + len(self.tokenizer.encode(generated_text[:rationale_section_start], add_special_tokens=False))
                    rationale_section_end_pos = prompt_len + len(self.tokenizer.encode(generated_text[:rationale_section_end], add_special_tokens=False))
                    
                    # Result section: from <result> to </result> (inclusive)
                    result_section_start = result_start
                    result_section_end = result_end + len('</result>')
                    result_section_start_pos = prompt_len + len(self.tokenizer.encode(generated_text[:result_section_start], add_special_tokens=False))
                    result_section_end_pos = prompt_len + len(self.tokenizer.encode(generated_text[:result_section_end], add_special_tokens=False))
               
                    if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                        print(f"Section parsing details:")
                        print(f"  rationale_section_start: {rationale_section_start}, rationale_section_end: {rationale_section_end}")
                        print(f"  rationale_section_start_pos: {rationale_section_start_pos}, rationale_section_end_pos: {rationale_section_end_pos}")
                        print(f"  result_section_start: {result_section_start}, result_section_end: {result_section_end}")
                        print(f"  result_section_start_pos: {result_section_start_pos}, result_section_end_pos: {result_section_end_pos}")
                        print(f"  rationale section length: {rationale_section_end_pos - rationale_section_start_pos} tokens")
                        print(f"  result section length: {result_section_end_pos - result_section_start_pos} tokens")
                    
                    # Supervise rationale section (including tags)
                    if rationale_section_start_pos < seq_len and rationale_section_end_pos <= seq_len:
                        rationale_section_end_pos = min(rationale_section_end_pos, seq_len)
                        labels[i, rationale_section_start_pos:rationale_section_end_pos] = input_ids[i, rationale_section_start_pos:rationale_section_end_pos]
                        loss_weights[i, rationale_section_start_pos:rationale_section_end_pos] = self.rationale_weight
                        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                            print(f"Supervised rationale section with weight {self.rationale_weight} at positions {rationale_section_start_pos}:{rationale_section_end_pos}")
                    else:
                        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                            print(f"Rationale section out of bounds: start={rationale_section_start_pos}, end={rationale_section_end_pos}, seq_len={seq_len}")
                        raise ValueError("rationale section not found")
                        
                    # Supervise result section (including tags)
                    if result_section_start_pos < seq_len and result_section_end_pos <= seq_len:
                        result_section_end_pos = min(result_section_end_pos, seq_len)
                        labels[i, result_section_start_pos:result_section_end_pos] = input_ids[i, result_section_start_pos:result_section_end_pos]
                        loss_weights[i, result_section_start_pos:result_section_end_pos] = self.answer_weight
                        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                            print(f"Supervised result section with weight {self.answer_weight} at positions {result_section_start_pos}:{result_section_end_pos}")
                    else:
                        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                            print(f"Result section out of bounds: start={result_section_start_pos}, end={result_section_end_pos}, seq_len={seq_len}")
                        raise ValueError("result section not found")
                    
                    if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                        print(f"\n--- SECTION SUPERVISION DEBUG ---")
                        print(f"Generated text: {generated_text[:200]}...")
                        print(f"Rationale section: pos {rationale_section_start_pos}-{rationale_section_end_pos}, weight {self.rationale_weight}")
                        print(f"Result section: pos {result_section_start_pos}-{result_section_end_pos}, weight {self.answer_weight}")
                        
                        # Show actual content being supervised (decode from tokens)
                        rationale_tokens = input_ids[i, rationale_section_start_pos:rationale_section_end_pos]
                        result_tokens = input_ids[i, result_section_start_pos:result_section_end_pos]
                        rationale_content = self.tokenizer.decode(rationale_tokens, skip_special_tokens=False)
                        result_content = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
                        print(f"Supervised rationale section: '{rationale_content[:100]}...'")
                        print(f"Supervised result section: '{result_content}'")
                        print(f"--- END SECTION SUPERVISION DEBUG ---\n")
                else:
                    # Fallback: if we can't parse the structure, use answer_weight for all (ensure supervision)
                    if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                        print(f"\n--- PARSING FAILED DEBUG ---")
                        print(f"Generated text: {generated_text[:200]}...")
                        print(f"Could not find rationale/result tags, using answer_weight={self.answer_weight} for all tokens")
                        print(f"--- END PARSING FAILED DEBUG ---\n")
                    loss_weights[i, prompt_len:seq_len] = self.answer_weight

        
        # Ensure padding is ignored in loss (safeguard)
        labels = labels.masked_fill(attention_mask == 0, -100)
        # Set loss weights to 0 for padding tokens
        loss_weights = loss_weights.masked_fill(labels == -100, 0.0)
        # Debug: Show final weight distribution
        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:
            print("=== FINAL DEBUG OUTPUT ===")
            print("Labels:", labels.tolist())
            print("Loss weights (full):", loss_weights.tolist())
            
            # Show weight distribution
            unique_weights, counts = torch.unique(loss_weights, return_counts=True)
            print("Weight distribution:")
            for weight, count in zip(unique_weights, counts):
                print(f"  Weight {weight.item()}: {count.item()} tokens")
            print("=== END FINAL DEBUG ===")

        # Always-on sanity checks and justifications (concise)
        try:
            bsz, seqlen = input_ids.shape
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            # 1) Padding tokens => attention_mask == 0
            if pad_id is not None:
                pad_mask = (input_ids == pad_id)
                if (attention_mask[pad_mask] != 0).any():
                    print("⚠️ [Warn] Padding tokens not fully masked in attention.")
            # 2) Loss ignores padding: attention_mask==0 => labels==-100
            if (labels[attention_mask == 0] != -100).any():
                print("⚠️ [Warn] Labels not -100 at padding positions.")
            # 3) Supervision span diagnostics
            sup_counts = [(labels[i] != -100).sum().item() for i in range(labels.size(0))]
        except Exception as e:
            print(f"⚠️ [Warn] Sanity checks failed with error: {e}")

        # Debug: Check label masking effectiveness
        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:
            print(f"\n--- LABEL MASKING DEBUG ---")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Check first sample
            sample_input_ids = input_ids[0]
            sample_labels = labels[0]
            
            # Count masked vs supervised tokens
            masked_tokens = (sample_labels == -100).sum().item()
            supervised_tokens = (sample_labels != -100).sum().item()
            total_tokens = len(sample_input_ids)
            
            print(f"Total tokens: {total_tokens}")
            print(f"Masked tokens (user input): {masked_tokens}")
            print(f"Supervised tokens (rationale): {supervised_tokens}")
            print(f"Mask ratio: {masked_tokens/total_tokens:.2%}")
            
            # Show the supervised token IDs and their text
            supervised_indices = (sample_labels != -100).nonzero(as_tuple=True)[0]
            if len(supervised_indices) > 0:
                supervised_token_ids = sample_input_ids[supervised_indices]
                supervised_token_text = [self.tokenizer.decode([tid]) for tid in supervised_token_ids]
                print(f"Supervised token IDs: {supervised_token_ids.tolist()}")
                print(f"Supervised token text: {supervised_token_text}")
                
                # Show first 10 and last 10 supervised tokens
                if len(supervised_token_ids) > 20:
                    print(f"First 10 supervised tokens: {supervised_token_ids[:10].tolist()}")
                    print(f"First 10 supervised text: {supervised_token_text[:10]}")
                    print(f"Last 10 supervised tokens: {supervised_token_ids[-10:].tolist()}")
                    print(f"Last 10 supervised text: {supervised_token_text[-10:]}")
                else:
                    print(f"All supervised tokens: {supervised_token_ids.tolist()}")
                    print(f"All supervised text: {supervised_token_text}")
            
            # Check user text length vs what was masked
            user_text = user_texts[0]
            user_ids = self.tokenizer(user_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            print(f"User text length: {len(user_text)} chars")
            print(f"User token count: {len(user_ids)}")
            print(f"User text preview: {user_text[:200]}...")
            
            print(f"--- END LABEL MASKING DEBUG ---\n")
        
        # Debug: Print tokenization details for first batch item
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:  # Only debug first 3 batches
            print(f"\n=== DEBUG BATCH {self._debug_count} ===")
            print(f"Full text: {full_texts[0][:200]}...")
            print(f"User text: {user_texts[0][:200]}...")
            
            # Debug: Show the difference between full_text and user_text
            full_text = full_texts[0]
            user_text = user_texts[0]
            print(f"Full text length: {len(full_text)} chars")
            print(f"User text length: {len(user_text)} chars")
            
            # Check if user_text is a prefix of full_text
            if full_text.startswith(user_text):
                print("✅ User text is a proper prefix of full text")
                assistant_text = full_text[len(user_text):]
                print(f"Assistant text: '{assistant_text}'")
            else:
                print("❌ User text is NOT a prefix of full text - this will cause masking issues!")
                print("This suggests the chat template is not working correctly.")
            
            # Get rationale from the batch item
            rationale = batch[0].get("rationale") if isinstance(batch[0], dict) else batch[0].rationale
            print(f"Rationale: '{rationale[:100] if rationale else 'None'}...'")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Show which tokens are supervised (not -100)
            supervised_mask = labels[0] != -100
            supervised_tokens = input_ids[0][supervised_mask]
            
            print(f"Supervised tokens: {supervised_tokens.tolist()}")
            print(f"Supervised token text: {[self.tokenizer.decode([tid]) for tid in supervised_tokens]}")
            print(f"Number of supervised tokens: {len(supervised_tokens)}")
            print(f"Expected: rationale tokens")
            
            print("=" * 50)

        # Keep tensors on CPU for Trainer to move; ensure standard dtypes
        pixel_values = pixel_values.to(dtype=torch.float32)
        if grid_thw is not None:
            if not isinstance(grid_thw, torch.Tensor):
                grid_thw = torch.tensor(grid_thw, dtype=torch.long)
            else:
                grid_thw = grid_thw.to(dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "loss_weights": loss_weights,
            "image_grid_thw": grid_thw,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./qwen-vl-sft-rationale")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_accum_steps", type=int, default=8)
    parser.add_argument("--num_video_frames", type=int, default=5)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--system_prompt", type=str, default=os.path.join("sft", "prompts", "sft_system.txt"))
    parser.add_argument("--user_prompt", type=str, default=os.path.join("sft", "prompts", "sft_user.txt"))
    parser.add_argument("--eval_system_prompt", type=str, default=None, help="System prompt for evaluation (auto-selected based on eval_use_rationale_format)")
    parser.add_argument("--eval_use_rationale_format", action="store_true", help="Use rationale format for evaluation (default: answer-only format)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", type=str, default="auto", choices=["auto", "4bit", "8bit", "bf16"])
    parser.add_argument("--resize_images", type=str, default="448x448", help="Resize images to WIDTHxHEIGHT (e.g., 224x224) to reduce memory usage")
    parser.add_argument("--debug_generation", action="store_true", help="Enable detailed generation logging during training")
    parser.add_argument("--debug_tokenization", action="store_true", help="Enable tokenization debugging during training")
    parser.add_argument("--rationale_weight", type=float, default=1.0, help="Loss weight for rationale tokens (default: 1.0)")
    parser.add_argument("--answer_weight", type=float, default=2.0, help="Loss weight for answer tokens (default: 2.0)")
    parser.add_argument("--enable_two_stage", action="store_true", help="Enable two-stage training: answer-only first, then rationale")
    parser.add_argument("--answer_only_epochs", type=int, default=5, help="Number of epochs for answer-only training stage (default: 5)")
    parser.add_argument("--lora_targets", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated LoRA target module names. Default includes attention + MLP.")
    parser.add_argument("--rationale_subdir", type=str, default="rationales", help="Subdirectory under each category holding rationale .txt files (e.g. 'rationales_claude_sonnet4', 'rationales_unstructured')")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="If > 0, save every N optimization steps (overrides epoch-level saving).")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Parse resize dimensions if provided
    resize_to = None
    if args.resize_images:
        try:
            width, height = map(int, args.resize_images.split('x'))
            resize_to = (width, height)
            print(f"[Info] Images will be resized to {width}x{height}")
        except ValueError:
            raise ValueError(f"Invalid resize format: {args.resize_images}. Use format like '224x224'")

    # Determine if we're doing two-stage training
    enable_two_stage = args.enable_two_stage
    answer_only_epochs = args.answer_only_epochs if enable_two_stage else 0
    rationale_epochs = args.epochs - answer_only_epochs if enable_two_stage else args.epochs
    
    print(f"[Info] Training configuration:")
    print(f"  Two-stage training: {enable_two_stage}")
    if enable_two_stage:
        print(f"  Answer-only epochs: {answer_only_epochs}")
        print(f"  Rationale epochs: {rationale_epochs}")
    else:
        print(f"  Total epochs: {args.epochs} (rationale training only)")

    # Build items based on training stage
    if enable_two_stage and answer_only_epochs > 0:
        print("[Info] Building answer-only items for first stage...")
        items = build_answer_only_items(args.dataset_dir, num_video_frames=args.num_video_frames, resize_to=resize_to)
        if not items:
            raise RuntimeError("No items found for answer-only training.")
        print(f"[Info] Loaded {len(items)} answer-only items.")
    else:
        print("[Info] Building items with rationales from dataset...")
        items = build_items(args.dataset_dir, num_video_frames=args.num_video_frames, resize_to=resize_to, rationale_subdir=args.rationale_subdir)
        if not items:
            raise RuntimeError("No items with rationales found. Run sft.generate_rationales first.")
        print(f"[Info] Loaded {len(items)} items with rationales.")

    # Split (train/val) by subcategory
    train_items, val_items = split_items_by_subcategory(items, val_ratio=0.1)

    print("[Info] Loading model & processor...")
    model, processor = build_model_and_processor(args.model_id, quant=args.quant)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.quant in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # LoRA targets — configurable via --lora_targets
    requested_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    available = set()
    for n, _ in model.named_modules():
        for t in requested_targets:
            if n.endswith(t):
                available.add(t)
    lora_targets = [t for t in requested_targets if t in available]
    if not lora_targets:
        # Fallback to legacy behavior
        for n, _ in model.named_modules():
            if any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
                break
        if not lora_targets:
            lora_targets = ["W_pack", "out_proj"]
    print(f"[Info] LoRA targets resolved: {lora_targets}")
    if set(requested_targets) - set(lora_targets):
        missing = sorted(set(requested_targets) - set(lora_targets))
        print(f"[Warn] Requested LoRA targets not found in model: {missing}")
    # LoRA will be applied by TRL's SFTTrainer via peft_config

    system_text = _read_text(args.system_prompt)
    user_template = _read_text(args.user_prompt)
    
    # Load evaluation system prompt (auto-select based on eval_use_rationale_format)
    eval_use_rationale_format = args.eval_use_rationale_format

    if eval_use_rationale_format:
        eval_system_prompt_path = os.path.join("sft", "prompts", "sft_system.txt")
    else:
        eval_system_prompt_path = os.path.join("sft", "prompts", "answer_only_system.txt")

    
    eval_system_text = _read_text(eval_system_prompt_path)

    # Create appropriate collator based on training stage
    if enable_two_stage and answer_only_epochs > 0:
        # First stage: answer-only training - use answer-only system prompt
        answer_only_system_prompt_path = os.path.join("sft", "prompts", "answer_only_system.txt")
        answer_only_system_text = _read_text(answer_only_system_prompt_path)
        collator = QwenVLAnswerOnlyCollator(
            processor=processor, tokenizer=tokenizer, device=device,
            system_text=answer_only_system_text, user_template=user_template, debug_tokenization=args.debug_tokenization
        )
    else:
        # Second stage or single-stage: rationale training - use rationale system prompt
        rationale_system_prompt_path = os.path.join("sft", "prompts", "sft_system.txt")
        rationale_system_text = _read_text(rationale_system_prompt_path)
        collator = QwenVLRationaleCollator(
            processor=processor, tokenizer=tokenizer, device=device,
            system_text=rationale_system_text, user_template=user_template, debug_tokenization=args.debug_tokenization,
            rationale_weight=args.rationale_weight, answer_weight=args.answer_weight
        )
    # Debug: Check dataset items
    if args.debug_tokenization:
        print(f"\n[DEBUG] Dataset loading debug:")
        print(f"Total train items: {len(train_items)}")
        print(f"Total val items: {len(val_items) if val_items else 0}")
        print(f"[DEBUG] Loss weights - Rationale: {args.rationale_weight}, Answer: {args.answer_weight}")
        
        # Check first few items
        for i in range(min(3, len(train_items))):
            item = train_items[i]
            print(f"Train item {i}: answer_letter='{item.answer_letter}', question='{item.question[:50]}...', rationale='{item.rationale[:50]}...'")
        
        if val_items:
            for i in range(min(3, len(val_items))):
                item = val_items[i]
                print(f"Val item {i}: answer_letter='{item.answer_letter}', question='{item.question[:50]}...', rationale='{item.rationale[:50]}...'")
    # Create datasets with appropriate validation based on training stage
    is_answer_only_stage = enable_two_stage and answer_only_epochs > 0
    train_ds = TorchSFTDataset(train_items, is_answer_only=is_answer_only_stage)
    eval_ds = TorchSFTDataset(val_items, is_answer_only=is_answer_only_stage) if val_items else None

    # Determine number of epochs for current stage
    current_epochs = answer_only_epochs if (enable_two_stage and answer_only_epochs > 0) else rationale_epochs
    
    use_step_strategy = args.save_steps and args.save_steps > 0
    sft_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.gradient_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=current_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to=[],
        dataloader_pin_memory=False,  # Avoid device conflicts
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    if use_step_strategy:
        sft_args_kwargs.update(
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=args.save_steps,
            save_steps=args.save_steps,
            save_total_limit=20,
        )
        print(f"[Info] Using step-based eval/save every {args.save_steps} steps")
    else:
        sft_args_kwargs.update(
            eval_strategy="epoch",
            save_strategy="epoch",
        )
    sft_args = TrainingArguments(**sft_args_kwargs)

    # Debug: Print trainer creation
    if args.debug_tokenization:
        print(f"\n[DEBUG] Creating SFTTrainer with debug_tokenization={args.debug_tokenization}")
        print(f"[DEBUG] Train dataset size: {len(train_ds)}")
        print(f"[DEBUG] Eval dataset size: {len(eval_ds) if eval_ds else 0}")
    
    # Debug: Check what fields SFTTrainer expects
    if args.debug_tokenization:
        print(f"\n[DEBUG] SFTTrainer configuration:")
        print(f"[DEBUG] Train dataset type: {type(train_ds)}")
        print(f"[DEBUG] Eval dataset type: {type(eval_ds)}")
        print(f"[DEBUG] Data collator type: {type(collator)}")
        
        # Test a single dataset item to see what SFTTrainer receives
        print(f"\n[DEBUG] Testing dataset item retrieval:")
        test_item = train_ds[0]
        print(f"[DEBUG] Dataset item 0: {test_item}")
        print(f"[DEBUG] Dataset item 0 keys: {list(test_item.keys())}")
        print(f"[DEBUG] Dataset item 0 types: {[(k, type(v)) for k, v in test_item.items()]}")

    # Create appropriate trainer based on training stage
    if enable_two_stage and answer_only_epochs > 0:
        # First stage: use standard SFTTrainer for answer-only training
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            peft_config=LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_targets,
            ),
        )
    else:
        # Second stage or single-stage: use WeightedSFTTrainer for rationale training
        trainer = WeightedSFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,  # Disable SFTTrainer's internal text field processing
            peft_config=LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_targets,
            ),
        )
    
    # Debug: Test DataLoader to see what it retrieves
    if args.debug_tokenization:
        print(f"\n[DEBUG] Testing DataLoader retrieval:")
        from torch.utils.data import DataLoader
        # Use our custom collator for the test DataLoader
        test_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collator)
        test_batch = next(iter(test_dataloader))
        print(f"[DEBUG] DataLoader batch keys: {list(test_batch.keys())}")
        print(f"[DEBUG] DataLoader batch types: {[(k, type(v)) for k, v in test_batch.items()]}")
    
    # Ensure model is on the correct device
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(device)

    if val_items:
        debug_dir = os.path.join(args.output_dir, "debug")
        eval_log_file = os.path.join(args.output_dir, "log.txt")
        callback = ValEvalCallback(processor, val_items, device, args.eval_limit, debug_dir, eval_system_text, user_template, True, eval_log_file, eval_use_rationale_format, train_items=None)
        callback.debug_generation = args.debug_generation
        trainer.add_callback(callback)
    
    # Add training accuracy callback
    train_acc_callback = TrainAccuracyCallback()
    trainer.add_callback(train_acc_callback)
    
    # Add epoch metrics logger callback
    epoch_logger = EpochMetricsLogger(args.output_dir)
    trainer.add_callback(epoch_logger)

    # Baseline eval
    debug_dir = os.path.join(args.output_dir, "debug")
    if val_items:
        print("[Eval] Baseline on val...")
        base_acc = evaluate_accuracy(model, processor, val_items, device, limit=args.eval_limit, debug_dir=debug_dir, system_text=eval_system_text, user_template=user_template, eval_plot=True, use_rationale_format=eval_use_rationale_format, eval_type="baseline")
        print(f"[Eval] Baseline val acc: {base_acc * 100:.2f}%")
        # No test eval per epoch

    # Save hyperparameters to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    hyperparams = {
        "model_id": args.model_id,
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accum_steps": args.gradient_accum_steps,
        "num_video_frames": args.num_video_frames,
        "eval_limit": args.eval_limit,
        "system_prompt": args.system_prompt,
        "user_prompt": args.user_prompt,
        "eval_system_prompt": args.eval_system_prompt,
        "eval_use_rationale_format": args.eval_use_rationale_format,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "quant": args.quant,
        "resize_images": args.resize_images,
        "debug_generation": args.debug_generation,
        "debug_tokenization": args.debug_tokenization,
        "rationale_weight": args.rationale_weight,
        "answer_weight": args.answer_weight,
        "enable_two_stage": args.enable_two_stage,
        "answer_only_epochs": args.answer_only_epochs,
        "train_items_count": len(train_items),
        "val_items_count": len(val_items) if val_items else 0,
        "total_items_count": len(items),
        "lora_targets": lora_targets,
        "device": str(device),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    hyperparams_file = os.path.join(args.output_dir, "hyperparameters.json")
    with open(hyperparams_file, "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2, ensure_ascii=False)
    print(f"[Save] Hyperparameters saved to {hyperparams_file}")

    # Two-stage training logic
    if enable_two_stage and answer_only_epochs > 0:
        print(f"\n{'='*60}")
        print(f"STAGE 1: ANSWER-ONLY TRAINING ({answer_only_epochs} epochs)")
        print(f"{'='*60}")
        
        # Stage 1: Answer-only training
        trainer.train()
        
        # Save intermediate checkpoint
        stage1_checkpoint_dir = os.path.join(args.output_dir, "stage1_answer_only")
        os.makedirs(stage1_checkpoint_dir, exist_ok=True)
        trainer.model.save_pretrained(stage1_checkpoint_dir)
        print(f"[Save] Stage 1 LoRA adapter saved to {stage1_checkpoint_dir}")
        
        # Stage 1 evaluation
        print("[Eval] Stage 1 (answer-only) eval...")
        if val_items:
            stage1_acc = evaluate_accuracy(model, processor, val_items, device, limit=args.eval_limit, debug_dir=debug_dir, system_text=eval_system_text, user_template=user_template, eval_plot=True, use_rationale_format=False, eval_type="stage1_answer_only")
            print(f"[Eval] Stage 1 val acc: {stage1_acc * 100:.2f}%")
        
        # Prepare for Stage 2: Rationale training
        if rationale_epochs > 0:
            print(f"\n{'='*60}")
            print(f"STAGE 2: RATIONALE TRAINING ({rationale_epochs} epochs)")
            print(f"{'='*60}")
            
            # Load rationale items for second stage
            print("[Info] Building items with rationales for second stage...")
            rationale_items = build_items(args.dataset_dir, num_video_frames=args.num_video_frames, resize_to=resize_to, rationale_subdir=args.rationale_subdir)
            if not rationale_items:
                raise RuntimeError("No items with rationales found for second stage. Run sft.generate_rationales first.")
            print(f"[Info] Loaded {len(rationale_items)} items with rationales for second stage.")
            
            # Split rationale items
            rationale_train_items, rationale_val_items = split_items_by_subcategory(rationale_items, val_ratio=0.1)
            
            # Create rationale datasets (second stage, not answer-only)
            rationale_train_ds = TorchSFTDataset(rationale_train_items, is_answer_only=False)
            rationale_eval_ds = TorchSFTDataset(rationale_val_items, is_answer_only=False) if rationale_val_items else None
            
            # Create rationale collator - use rationale system prompt for second stage
            rationale_system_prompt_path = os.path.join("sft", "prompts", "sft_system.txt")
            rationale_system_text = _read_text(rationale_system_prompt_path)
            rationale_collator = QwenVLRationaleCollator(
                processor=processor, tokenizer=tokenizer, device=device,
                system_text=rationale_system_text, user_template=user_template, debug_tokenization=args.debug_tokenization,
                rationale_weight=args.rationale_weight, answer_weight=args.answer_weight
            )
            
            # Create rationale training arguments
            rationale_sft_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=max(1, args.batch_size),
                gradient_accumulation_steps=args.gradient_accum_steps,
                learning_rate=args.lr,
                num_train_epochs=rationale_epochs,
                warmup_ratio=args.warmup_ratio,
                lr_scheduler_type="cosine",
                logging_steps=10,
                bf16=torch.cuda.is_available(),
                report_to=[],
                dataloader_pin_memory=False,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                remove_unused_columns=False
            )
            
            # Create rationale trainer
            rationale_trainer = WeightedSFTTrainer(
                model=model,
                args=rationale_sft_args,
                train_dataset=rationale_train_ds,
                eval_dataset=rationale_eval_ds,
                data_collator=rationale_collator,
                peft_config=LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_targets,
                ),
            )
            
            # Add callbacks for rationale training
            if rationale_val_items:
                rationale_debug_dir = os.path.join(args.output_dir, "debug")
                rationale_eval_log_file = os.path.join(args.output_dir, "log.txt")
                rationale_callback = ValEvalCallback(processor, rationale_val_items, device, args.eval_limit, rationale_debug_dir, eval_system_text, user_template, True, rationale_eval_log_file, eval_use_rationale_format, train_items=None)
                rationale_callback.debug_generation = args.debug_generation
                rationale_trainer.add_callback(rationale_callback)
            
            # Add training accuracy callback
            rationale_train_acc_callback = TrainAccuracyCallback()
            rationale_trainer.add_callback(rationale_train_acc_callback)
            
            # Add epoch metrics logger callback
            rationale_epoch_logger = EpochMetricsLogger(args.output_dir), 
            rationale_trainer.add_callback(rationale_epoch_logger)
            
            # Stage 2: Rationale training
            rationale_trainer.train()
            
            # Save final model
            rationale_trainer.model.save_pretrained(args.output_dir)
            print(f"[Save] Final LoRA adapter saved to {args.output_dir}")
            
            # Final evaluation
            print("[Eval] Final eval...")
            if rationale_val_items:
                final_acc = evaluate_accuracy(model, processor, rationale_val_items, device, limit=args.eval_limit, debug_dir=rationale_debug_dir, system_text=eval_system_text, user_template=user_template, eval_plot=True, use_rationale_format=eval_use_rationale_format, eval_type="final")
                print(f"[Eval] Final val acc: {final_acc * 100:.2f}%")
        else:
            print("[Info] No rationale epochs specified, skipping second stage.")
    else:
        # Single-stage training (original behavior)
        print(f"\n{'='*60}")
        print(f"SINGLE-STAGE: RATIONALE TRAINING ({args.epochs} epochs)")
        print(f"{'='*60}")
        
        trainer.train()

        trainer.model.save_pretrained(args.output_dir)
        print(f"[Save] LoRA adapter saved to {args.output_dir}")

        print("[Eval] Final eval...")
        if val_items:
            final_acc = evaluate_accuracy(model, processor, val_items, device, limit=args.eval_limit, debug_dir=debug_dir, system_text=eval_system_text, user_template=user_template, eval_plot=True, use_rationale_format=eval_use_rationale_format, eval_type="final")
            print(f"[Eval] Final val acc: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()


