

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

from dataset.causal_vl import CausalVLDataset

# Import shared evaluation utilities
from .eval_utils import evaluate_accuracy, ValEvalCallback, _read_text, build_model_and_processor, split_items_by_subcategory


CHOICE_LETTERS = ["A", "B", "C", "D"]


# Import shared types
from .shared_types import SFTItemAnswerOnly as SFTItem


def _open_image(p: str, resize_to: Optional[tuple] = None) -> Optional[Image.Image]:
    try:
        img = Image.open(p).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


def build_items(dataset_root: str, num_video_frames: int, max_images: int = 8, resize_to: Optional[tuple] = None) -> List[SFTItem]:
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
        items.append(SFTItem(images=imgs, question=q, answer_letter=gt, category=s.category, subcategory=s.subcategory, index=item_index, ann_path=s.ann_path))
        item_index += 1
    
    print(f"[Info] Built {len(items)} items, skipped {skipped_count} invalid samples")
    return items


class TorchSFTDataset(Dataset):
    def __init__(self, items: List[SFTItem]):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        # Return a dict so TRL's SFTTrainer can introspect sample keys safely
        # Add validation to catch corrupted items
        if it.question is None or it.answer_letter is None or not it.images:
            raise ValueError(f"⚠️  Corrupted item at index {idx}: question={it.question is not None}, answer_letter={it.answer_letter is not None}, images={len(it.images) if it.images else 0}")
        
        # Return structured fields for cleaner processing
        result = {
            "images": it.images,
            "question": it.question,
            "final_answer": it.answer_letter,
        }
        return result


class QwenVLAnswerOnlyCollator:
    def __init__(self, processor, tokenizer, device: torch.device, system_text: str, user_template: str, debug_tokenization: bool = False):
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.system_text = system_text
        self.user_template = user_template
        self.debug_tokenization = debug_tokenization
        
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
            answer_letter = it.get("final_answer") if isinstance(it, dict) else it.final_answer
            
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
                    {"type": "text", "text": answer_letter}
                ]},
            ]
            full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Debug: Print what we extracted from the batch item
            if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:
                print(f"🔍 Extracted from batch item: question={question is not None}, answer_letter={answer_letter is not None}, images={len(images) if images else 0}")
                print(f"🔍 Batch item keys: {list(it.keys()) if isinstance(it, dict) else 'not a dict'}")
                print(f"🔍 Batch item type: {type(it)}")
            
            # Debug: Check if dataset items are corrupted
            if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3:
                print(f"\n--- DATASET ITEM DEBUG ---")
                print(f"Question: {question[:100] if question else 'None'}")
                print(f"Answer letter: '{answer_letter}'")
                print(f"Images count: {len(images) if images else 0}")
                print(f"Item type: {type(it)}")
                if isinstance(it, dict):
                    print(f"Item keys: {list(it.keys())}")
                print(f"--- END DATASET ITEM DEBUG ---\n")
            
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
            padding=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        pixel_values = enc["pixel_values"]
        grid_thw = enc.get("image_grid_thw", None)

        # Supervise ONLY the final answer token:
        # - Compute exact prompt length using the same processor (text + images)
        # - Search for the answer token ONLY within the generated region (after prompt, before padding)
        # - Mask everything to -100 except the first occurrence of the answer token in that region
        labels = input_ids.clone()
        labels.fill_(-100)
        for i, (ut, imgs) in enumerate(zip(user_texts, images_per)):
            enc_prompt = self.processor(text=[ut], images=[imgs], return_tensors="pt", padding=False)
            prompt_len = enc_prompt["input_ids"].shape[1]
            # Effective sequence length ignoring padding
            seq_len = int(attention_mask[i].sum().item())

            # Determine the single-token answer id (e.g., A/B/C/D)
            ans_text = batch[i].get("final_answer") if isinstance(batch[i], dict) else getattr(batch[i], "answer_letter", None)
            if not ans_text:
                continue
            ans_ids = self.tokenizer(ans_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if len(ans_ids) != 1:
                # If it doesn't tokenize to a single id, skip supervision for this sample
                continue
            ans_id = ans_ids[0]

            # Search within generated region only: [prompt_len : seq_len)
            if prompt_len < seq_len:
                search_slice = input_ids[i, prompt_len:seq_len]
                positions = (search_slice == ans_id).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    # Supervise the first occurrence in generated region
                    pos = prompt_len + positions[0].item()
                    labels[i, pos] = ans_id

            if self.debug_tokenization and hasattr(self, '_debug_count') and self._debug_count <= 3 and i == 0:
                print(f"\n--- ANSWER TOKEN SUPERVISION DEBUG ---")
                print(f"Prompt_len: {prompt_len}, seq_len: {seq_len}")
                print(f"Answer text: {ans_text}, token id: {int(ans_id)}")
                if prompt_len < seq_len:
                    print(f"Generated region ids (trunc): {input_ids[i, prompt_len:min(seq_len, prompt_len+50)].tolist()}")
                supervised_positions = (labels[i] != -100).nonzero(as_tuple=True)[0].tolist()
                print(f"Supervised positions: {supervised_positions}")
                print(f"--- END ANSWER TOKEN SUPERVISION DEBUG ---\n")
        
        # Ensure padding is ignored in loss (safeguard)
        labels = labels.masked_fill(attention_mask == 0, -100)

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
            # 3) Supervision cardinality and match
            for i in range(bsz):
                sup_idx = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if sup_idx.numel() == 0:
                    # ok when we intentionally skipped (e.g., multi-token answer)
                    continue
                if sup_idx.numel() > 1:
                    print(f"⚠️ [Warn] Multiple supervised tokens in sample {i}: {sup_idx.tolist()}")
                # If final_answer available, check match
                ans_text = batch[i].get("final_answer") if isinstance(batch[i], dict) else getattr(batch[i], "answer_letter", None)
                if ans_text:
                    ans_ids = self.tokenizer(ans_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                    if len(ans_ids) == 1:
                        ans_id = int(ans_ids[0])
                        if int(input_ids[i, sup_idx[-1]]) != ans_id:
                            dec_sup = self.tokenizer.decode([int(input_ids[i, sup_idx[-1]])])
                            print(f"⚠️ [Warn] Supervised token mismatch in sample {i}: got '{dec_sup}', expected '{ans_text}'.")
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
            print(f"Supervised tokens (assistant response): {supervised_tokens}")
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
            
            # Get answer letter from the batch item
            answer_letter = batch[0].get("final_answer") if isinstance(batch[0], dict) else batch[0].answer_letter
            print(f"Answer letter: '{answer_letter}'")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Show which tokens are supervised (not -100)
            supervised_mask = labels[0] != -100
            supervised_tokens = input_ids[0][supervised_mask]
            
            print(f"Supervised tokens: {supervised_tokens.tolist()}")
            print(f"Supervised token text: {[self.tokenizer.decode([tid]) for tid in supervised_tokens]}")
            print(f"Number of supervised tokens: {len(supervised_tokens)}")
            print(f"Expected: just the answer letter '{answer_letter}'")
            
            # Check if the supervised tokens match the expected answer
            if answer_letter and answer_letter != 'None':
                # Tokenize the expected answer
                expected_tokens = self.tokenizer(answer_letter, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                expected_text = [self.tokenizer.decode([tid]) for tid in expected_tokens]
                print(f"Expected answer tokens: {expected_tokens.tolist()}")
                print(f"Expected answer text: {expected_text}")
                
                # Check if supervised tokens end with expected answer
                if len(supervised_tokens) >= len(expected_tokens):
                    supervised_end = supervised_tokens[-len(expected_tokens):]
                    if torch.equal(supervised_end, expected_tokens):
                        print("✅ Supervised tokens correctly end with expected answer")
                    else:
                        print("❌ Supervised tokens do NOT end with expected answer")
                        print(f"Supervised end: {supervised_end.tolist()}")
                        print(f"Expected: {expected_tokens.tolist()}")
                else:
                    print("❌ Supervised tokens are shorter than expected answer")
            else:
                print("⚠️  Answer letter is 'None' or empty - this is a dataset issue!")
            
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./qwen-vl-sft-answer-only")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_accum_steps", type=int, default=8)
    parser.add_argument("--num_video_frames", type=int, default=5)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--system_prompt", type=str, default=os.path.join("sft", "prompts", "answer_only_system.txt"))
    parser.add_argument("--user_prompt", type=str, default=os.path.join("sft", "prompts", "sft_user.txt"))
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", type=str, default="auto", choices=["auto", "4bit", "8bit", "bf16"])
    parser.add_argument("--resize_images", type=str, default="448x448", help="Resize images to WIDTHxHEIGHT (e.g., 224x224) to reduce memory usage")
    parser.add_argument("--debug_generation", action="store_true", help="Enable detailed generation logging during training")
    parser.add_argument("--debug_tokenization", action="store_true", help="Enable tokenization debugging during training")
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

    print("[Info] Building items from dataset...")
    items = build_items(args.dataset_dir, num_video_frames=args.num_video_frames, resize_to=resize_to)
    if not items:
        raise RuntimeError("No items found in dataset.")
    print(f"[Info] Loaded {len(items)} items.")

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

    # LoRA targets
    lora_targets = []
    for n, _ in model.named_modules():
        if any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            break
    if not lora_targets:
        lora_targets = ["W_pack", "out_proj"]
    # LoRA will be applied by TRL's SFTTrainer via peft_config

    system_text = _read_text(args.system_prompt)
    user_template = _read_text(args.user_prompt)

    collator = QwenVLAnswerOnlyCollator(
        processor=processor, tokenizer=tokenizer, device=device,
        system_text=system_text, user_template=user_template, debug_tokenization=args.debug_tokenization
    )
    # Debug: Check dataset items
    if args.debug_tokenization:
        print(f"\n[DEBUG] Dataset loading debug:")
        print(f"Total train items: {len(train_items)}")
        print(f"Total val items: {len(val_items) if val_items else 0}")
        
        # Check first few items
        for i in range(min(3, len(train_items))):
            item = train_items[i]
            print(f"Train item {i}: answer_letter='{item.answer_letter}', question='{item.question[:50]}...'")
        
        if val_items:
            for i in range(min(3, len(val_items))):
                item = val_items[i]
                print(f"Val item {i}: answer_letter='{item.answer_letter}', question='{item.question[:50]}...'")
    
    train_ds = TorchSFTDataset(train_items)
    eval_ds = TorchSFTDataset(val_items) if val_items else None

    sft_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.gradient_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to=[],
        dataloader_pin_memory=False,  # Avoid device conflicts
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_token_accuracy",
        greater_is_better=True,
        remove_unused_columns=False
    )

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

    trainer = SFTTrainer(
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
        callback = ValEvalCallback(processor, val_items, device, args.eval_limit, debug_dir, system_text, user_template, True, eval_log_file, False, train_items)
        callback.debug_generation = args.debug_generation
        trainer.add_callback(callback)

    # Baseline eval
    debug_dir = os.path.join(args.output_dir, "debug")
    if val_items:
        print("[Eval] Baseline on val...")
        base_acc = evaluate_accuracy(model, processor, val_items, device, limit=args.eval_limit, debug_dir=debug_dir, system_text=system_text, user_template=user_template, eval_plot=True, eval_type="baseline")
        print(f"[Eval] Baseline val acc: {base_acc * 100:.2f}%")
    trainer.train()
        # No test eval per epoch

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    print(f"[Save] LoRA adapter saved to {args.output_dir}")

    print("[Eval] Final eval...")
    if val_items:
        final_acc = evaluate_accuracy(model, processor, val_items, device, limit=args.eval_limit, debug_dir=debug_dir, system_text=system_text, user_template=user_template, eval_plot=True, eval_type="final")
        print(f"[Eval] Final val acc: {final_acc * 100:.2f}%")
    # No final test eval


if __name__ == "__main__":
    main()