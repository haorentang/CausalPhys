"""
Shared evaluation utilities for SFT training scripts.
Contains common functions used by both answer-only and rationale training.
"""

import os
import re
import random
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import torch
from tqdm import tqdm
from transformers import AutoProcessor, trainer_pt_utils
from transformers.trainer_callback import TrainerCallback
from .shared_types import SFTItem, SFTItemAnswerOnly  # Import the SFTItem dataclasses


def _read_text(path: str) -> str:
    """Read text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _create_eval_debug_dir(base_debug_dir: str, eval_type: str = "evaluation") -> str:
    """
    Create a timestamped subfolder under the debug directory for each evaluation run.
    
    Args:
        base_debug_dir: Base debug directory path
        eval_type: Type of evaluation (e.g., "evaluation", "baseline", "final")
    
    Returns:
        Path to the created evaluation subfolder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(base_debug_dir, f"{eval_type}_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def _save_evaluation_summary(eval_dir: str, accuracy: float, category_stats: Dict, 
                           subcategory_stats: Dict, total_items: int, correct_items: int,
                           eval_type: str = "evaluation") -> None:
    """
    Save evaluation summary to a JSON file in the evaluation directory.
    
    Args:
        eval_dir: Directory to save the summary
        accuracy: Overall accuracy
        category_stats: Per-category statistics
        subcategory_stats: Per-subcategory statistics
        total_items: Total number of items evaluated
        correct_items: Number of correct predictions
        eval_type: Type of evaluation
    """
    summary = {
        "evaluation_type": eval_type,
        "timestamp": datetime.now().isoformat(),
        "overall_accuracy": accuracy,
        "total_items": total_items,
        "correct_items": correct_items,
        "category_accuracy": {
            cat: {
                "accuracy": stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                "correct": stats['correct'],
                "total": stats['total']
            }
            for cat, stats in category_stats.items()
        },
        "subcategory_accuracy": {
            subcat: {
                "accuracy": stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                "correct": stats['correct'],
                "total": stats['total']
            }
            for subcat, stats in subcategory_stats.items()
        }
    }
    
    summary_file = os.path.join(eval_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def split_items_by_subcategory(items: List[Union[SFTItem, SFTItemAnswerOnly]], 
                              val_ratio: float = 0.1) -> tuple:
    """
    Split items by subcategory, ensuring each subcategory has its own train/val split.
    
    Args:
        items: List of all items
        val_ratio: Ratio of items to use for validation (default 0.1)
    
    Returns:
        Tuple of (train_items, val_items)
    """
    # Group items by subcategory
    subcategory_items = {}
    for item in items:
        subcat = item.subcategory
        if subcat not in subcategory_items:
            subcategory_items[subcat] = []
        subcategory_items[subcat].append(item)
    
    train_items = []
    val_items = []
    
    print(f"[Info] Splitting {len(items)} items across {len(subcategory_items)} subcategories:")
    
    for subcat, subcat_items in subcategory_items.items():
        # Shuffle items within each subcategory
        random.shuffle(subcat_items)
        
        # Calculate split size for this subcategory
        val_size = max(1, int(len(subcat_items) * val_ratio)) if len(subcat_items) >= 10 else 0
        
        if val_size > 0:
            subcat_val_items = subcat_items[:val_size]
            subcat_train_items = subcat_items[val_size:]
        else:
            subcat_val_items = []
            subcat_train_items = subcat_items
        
        train_items.extend(subcat_train_items)
        val_items.extend(subcat_val_items)
        
        print(f"  {subcat}: {len(subcat_items)} total -> {len(subcat_train_items)} train, {len(subcat_val_items)} val")
    
    print(f"[Info] Final split: {len(train_items)} training instances, {len(val_items)} validation instances")
    return train_items, val_items


def build_model_and_processor(model_id: str, quant: str = "bf16"):
    """Build model and processor for training."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    import os
    
    # Check if model_id is a LoRA checkpoint path
    is_lora_checkpoint = False
    base_model_id = model_id
    
    if os.path.exists(model_id) and os.path.isfile(os.path.join(model_id, "adapter_config.json")):
        is_lora_checkpoint = True
        # Read the base model from adapter config
        import json
        with open(os.path.join(model_id, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
            base_model_id = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2-VL-7B-Instruct")
        print(f"[Info] Detected LoRA checkpoint at {model_id}")
        print(f"[Info] Base model: {base_model_id}")
    
    # Load base model
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    # Load LoRA adapter if it's a checkpoint
    if is_lora_checkpoint:
        print(f"[Info] Loading LoRA adapter from {model_id}")
        model = PeftModel.from_pretrained(model, model_id)
        print(f"[Info] LoRA adapter loaded successfully")
    
    # Load processor from base model
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    return model, processor


@torch.no_grad()
def _extract_choice(text: str, use_rationale_format: bool = False) -> Optional[str]:
    """
    Extract answer choice from generated text.
    
    Args:
        text: Generated text from model
        use_rationale_format: If True, look for <result>A</result> format.
                             If False, look for standalone A-D letters.
    
    Returns:
        Extracted choice (A, B, C, or D) or None if not found
    """
    import re as _re
    
    if use_rationale_format:
        # For rationale format: look for <result>A</result>
        m = _re.search(r"<result>([\s\S]*?)</result>", text, flags=_re.IGNORECASE)
        if m:
            inside = m.group(1)
            m2 = _re.search(r"([ABCD])", inside, flags=_re.IGNORECASE)
            if m2:
                return m2.group(1).upper()
        # Fallback: use last standalone A-D letter if tag missing
        ms = _re.findall(r"\b([ABCD])\b", text, flags=_re.IGNORECASE)
        return ms[-1].upper() if ms else None
    else:
        # For answer-only format: look for standalone A-D letters
        ms = _re.findall(r"\b([ABCD])\b", text, flags=_re.IGNORECASE)
        return ms[-1].upper() if ms else None


@torch.no_grad()
def _extract_rationale_and_result(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract both rationale and result from generated text.
    
    Args:
        text: Generated text from model
    
    Returns:
        Tuple of (rationale_text, result_choice) where:
        - rationale_text: The text between <rationale> and </rationale> tags
        - result_choice: The choice letter (A, B, C, or D) between <result> and </result> tags
    """
    import re as _re
    
    # Extract rationale
    rationale_match = _re.search(r"<rationale>([\s\S]*?)</rationale>", text, flags=_re.IGNORECASE)
    rationale_text = rationale_match.group(1).strip() if rationale_match else None
    
    # Extract result
    result_match = _re.search(r"<result>([\s\S]*?)</result>", text, flags=_re.IGNORECASE)
    if result_match:
        inside = result_match.group(1)
        choice_match = _re.search(r"([ABCD])", inside, flags=_re.IGNORECASE)
        result_choice = choice_match.group(1).upper() if choice_match else None
    else:
        result_choice = None
    
    return rationale_text, result_choice


def select_consistent_eval_samples(items: List[Union[SFTItem, SFTItemAnswerOnly]], 
                                 samples_per_category: int = 3) -> List[Union[SFTItem, SFTItemAnswerOnly]]:
    """
    Select consistent evaluation samples using predefined indices.
    
    Args:
        items: List of all evaluation items
        samples_per_category: Number of samples to select from each subcategory
    
    Returns:
        List of selected items for consistent evaluation across epochs
    """
    # Group items by subcategory first to discover available subcategories
    subcategory_items = {}
    for item in items:
        subcat = item.subcategory
        if subcat not in subcategory_items:
            subcategory_items[subcat] = []
        subcategory_items[subcat].append(item)
    
    # Dynamically create consistent indices for each discovered subcategory
    # Select the first N samples from each subcategory for consistent evaluation
    consistent_indices = {}
    total_selected = 0
    for subcat, subcat_items in subcategory_items.items():
        # Take up to samples_per_category items from each subcategory
        max_samples = min(samples_per_category, len(subcat_items))
        consistent_indices[subcat] = list(range(max_samples))
        total_selected += max_samples
        print(f"  {subcat}: {max_samples} items selected from {len(subcat_items)} available")
    
    print(f"Selected {total_selected} consistent evaluation samples from {len(subcategory_items)} subcategories")
    
    # Select samples based on consistent indices
    selected_items = []
    for subcategory, indices in consistent_indices.items():
        if subcategory in subcategory_items:
            subcat_items = subcategory_items[subcategory]
            # Sort by index to ensure consistent ordering
            subcat_items.sort(key=lambda x: x.index)
            
            # Select items at the specified indices (if they exist)
            for idx in indices[:samples_per_category]:
                if idx < len(subcat_items):
                    selected_items.append(subcat_items[idx])
    
    print(f"Selected {len(selected_items)} consistent evaluation samples from {len(consistent_indices)} subcategories")
    for subcategory, indices in consistent_indices.items():
        if subcategory in subcategory_items:
            available = len(subcategory_items[subcategory])
            selected = min(len(indices), samples_per_category, available)
            print(f"  {subcategory}: {available} items available, {selected} selected")
        else:
            print(f"  {subcategory}: 0 items available, 0 selected")
    
    return selected_items


@torch.no_grad()
def evaluate_accuracy(
    model,
    processor: AutoProcessor,
    items: List[Union[SFTItem, SFTItemAnswerOnly]],
    device: torch.device,
    limit: Optional[int] = None,
    debug_dir: Optional[str] = None,
    system_text: str = "",
    user_template: str = "{question}\n\nAnswer with only a single capital letter: A, B, C, or D.",
    eval_plot: bool = False,
    use_rationale_format: bool = False,
    use_consistent_samples: bool = True,
    samples_per_category: int = 3,
    eval_type: str = "evaluation",
) -> float:
    """
    Evaluate model accuracy on a list of items.
    
    Args:
        model: The model to evaluate
        processor: The processor for the model
        items: List of SFTItem to evaluate on
        device: Device to run evaluation on
        limit: Optional limit on number of items to evaluate
        debug_dir: Optional base directory to save debug outputs (creates timestamped subfolder)
        system_text: System prompt text
        user_template: User prompt template
        eval_plot: Whether to plot evaluation results
        use_rationale_format: Whether to use rationale format for choice extraction
        use_consistent_samples: Whether to use consistent sample selection across epochs
        samples_per_category: Number of samples to select from each subcategory
        eval_type: Type of evaluation (e.g., "evaluation", "baseline", "final")
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    
    model.eval()
    correct = 0
    total = 0
    
    # Track per-subcategory and per-category accuracy
    subcategory_stats = {}
    category_stats = {}
    
    # Create structured debug directory if debug_dir is provided
    eval_debug_dir = None
    if debug_dir:
        eval_debug_dir = _create_eval_debug_dir(debug_dir, eval_type)
        print(f"Created evaluation debug directory: {eval_debug_dir}")
    
    # Always evaluate on all items (or limited items)
    eval_items = items[:limit] if limit else items
    
    # Select consistent samples for debug output only
    debug_samples = None
    if use_consistent_samples and eval_debug_dir:
        debug_samples = select_consistent_eval_samples(items, samples_per_category)
        # Create a mapping of selected samples for quick lookup
        debug_sample_questions = {sample.question: sample for sample in debug_samples}
    
    for idx, it in enumerate(tqdm(eval_items, desc="Evaluating", total=len(eval_items))):
        try:
            # Build user content with images and question
            header_lines = [f"Question:\n{it.question.strip()}"]
            if it.images:
                header_lines.append(f"[Images provided: {len(it.images)}; consume in the given order as a sequence]")
            header = "\n\n".join(header_lines)

            user_content = []
            for img in it.images:
                user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": header})

            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_content},
            ]

            # Generate response
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = processor(text=[text], images=[it.images], return_tensors="pt")
            enc = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}

            with torch.no_grad():
                generated_full = model.generate(
                    **enc,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            # Extract generated text
            prompt_len = enc["input_ids"].shape[1]
            full_seq_ids = generated_full[0]
            gen_ids_only = full_seq_ids[prompt_len:]
            out_text = processor.decode(gen_ids_only, skip_special_tokens=True)
            
            pred = _extract_choice(out_text, use_rationale_format=use_rationale_format)
            is_correct = pred == it.answer_letter

            if is_correct:
                correct += 1
            total += 1
            
            # Track per-subcategory accuracy
            subcat = it.subcategory
            if subcat not in subcategory_stats:
                subcategory_stats[subcat] = {'correct': 0, 'total': 0}
            if is_correct:
                subcategory_stats[subcat]['correct'] += 1
            subcategory_stats[subcat]['total'] += 1
            
            # Track per-category accuracy
            cat = it.category
            if cat not in category_stats:
                category_stats[cat] = {'correct': 0, 'total': 0}
            if is_correct:
                category_stats[cat]['correct'] += 1
            category_stats[cat]['total'] += 1
            
            # Print detailed output for monitoring (first 5 examples)
            if idx < 5:
                print(f"--- Example {idx + 1} ---")
                print(f"Question: {it.question[:100]}...")
                print(f"Expected: {it.answer_letter}")
                print(f"Generated: {out_text[:100]}...")
                print(f"Predicted: {pred}")
                print(f"Correct: {is_correct}")
                print()
            
            # Save debug outputs only for selected consistent samples with subcategory organization
            if eval_debug_dir and debug_samples and it.question in debug_sample_questions:
                # Create subcategory subfolder
                subcat_dir = os.path.join(eval_debug_dir, "samples", subcat)
                os.makedirs(subcat_dir, exist_ok=True)
                
                # Find the index of this sample in the debug_samples list
                debug_idx = next(i for i, sample in enumerate(debug_samples) if sample.question == it.question)
                debug_file = os.path.join(subcat_dir, f"sample_{debug_idx:03d}.txt")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"Question: {it.question}\n")
                    f.write(f"Expected: {it.answer_letter}\n")
                    f.write(f"Generated: {out_text}\n")
                    f.write(f"Predicted: {pred}\n")
                    f.write(f"Correct: {is_correct}\n")
                    f.write(f"Category: {it.category}\n")
                    f.write(f"Subcategory: {it.subcategory}\n")
                    f.write(f"Index: {it.index}\n")
                    
        except Exception as e:
            print(f"Error evaluating item {idx}: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Save evaluation summary if debug directory is provided
    if eval_debug_dir:
        _save_evaluation_summary(eval_debug_dir, accuracy, category_stats, 
                               subcategory_stats, total, correct, eval_type)
        print(f"Evaluation summary saved to: {eval_debug_dir}/evaluation_summary.json")
    
    # Print per-category and per-subcategory accuracy
    print(f"\n=== Evaluation Results ===")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"\nPer-Category Accuracy:")
    for cat, stats in sorted(category_stats.items()):
        cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {cat}: {cat_accuracy:.3f} ({stats['correct']}/{stats['total']})")
    print(f"\nPer-Subcategory Accuracy:")
    for subcat, stats in sorted(subcategory_stats.items()):
        subcat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {subcat}: {subcat_accuracy:.3f} ({stats['correct']}/{stats['total']})")
    print("=" * 30)
    
    return accuracy


# @torch.no_grad()
# def evaluate_rationale_and_result_accuracy(
#     model,
#     processor: AutoProcessor,
#     items: List[Union[SFTItem, SFTItemAnswerOnly]],
#     device: torch.device,
#     limit: Optional[int] = None,
#     debug_dir: Optional[str] = None,
#     system_text: str = "",
#     user_template: str = "{question}\n\nAnswer with only a single capital letter: A, B, C, or D.",
#     eval_plot: bool = False,
#     use_rationale_format: bool = False,
#     eval_type: str = "rationale_evaluation",
# ) -> tuple[float, float]:
#     """
#     Evaluate model accuracy on rationale and result separately.
    
#     Args:
#         model: The model to evaluate
#         processor: The processor for the model
#         items: List of SFTItem to evaluate on
#         device: Device to run evaluation on
#         limit: Optional limit on number of items to evaluate
#         debug_dir: Optional base directory to save debug outputs (creates timestamped subfolder)
#         system_text: System prompt text
#         user_template: User prompt template
#         eval_plot: Whether to plot evaluation results
#         use_rationale_format: Whether to use rationale format for choice extraction
#         eval_type: Type of evaluation (e.g., "rationale_evaluation")
    
#     Returns:
#         Tuple of (rationale_accuracy, result_accuracy) as floats between 0 and 1
#     """
#     model.eval()
#     rationale_correct = 0
#     result_correct = 0
#     total = 0
    
#     # Create structured debug directory if debug_dir is provided
#     eval_debug_dir = None
#     if debug_dir:
#         eval_debug_dir = _create_eval_debug_dir(debug_dir, eval_type)
#         print(f"Created rationale evaluation debug directory: {eval_debug_dir}")
    
#     # Limit items if specified
#     eval_items = items[:limit] if limit else items
    
#     for idx, it in enumerate(eval_items):
#         try:
#             # Build user content with images and question
#             header_lines = [f"Question:\n{it.question.strip()}"]
#             if it.images:
#                 header_lines.append(f"[Images provided: {len(it.images)}; consume in the given order as a sequence]")
#             header = "\n\n".join(header_lines)

#             user_content = []
#             for img in it.images:
#                 user_content.append({"type": "image", "image": img})
#             user_content.append({"type": "text", "text": header})

#             messages = [
#                 {"role": "system", "content": system_text},
#                 {"role": "user", "content": user_content},
#             ]

#             # Generate response
#             text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             enc = processor(text=[text], images=[it.images], return_tensors="pt")
#             enc = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}

#             with torch.no_grad():
#                 generated_full = model.generate(
#                     **enc,
#                     max_new_tokens=512,
#                     do_sample=False,
#                     pad_token_id=processor.tokenizer.pad_token_id,
#                     eos_token_id=processor.tokenizer.eos_token_id,
#                 )

#             # Extract generated text
#             prompt_len = enc["input_ids"].shape[1]
#             full_seq_ids = generated_full[0]
#             gen_ids_only = full_seq_ids[prompt_len:]
#             out_text = processor.decode(gen_ids_only, skip_special_tokens=True)
            
#             # Extract rationale and result separately
#             pred_rationale, pred_result = _extract_rationale_and_result(out_text)
            
#             # Check rationale correctness (if we have ground truth rationale)
#             rationale_correct_this = False
#             if hasattr(it, 'rationale') and it.rationale and pred_rationale:
#                 # Simple similarity check - could be improved with more sophisticated metrics
#                 rationale_correct_this = len(pred_rationale.strip()) > 10  # Basic check for non-empty rationale
#             elif pred_rationale:
#                 rationale_correct_this = len(pred_rationale.strip()) > 10  # Basic check for non-empty rationale
            
#             # Check result correctness
#             result_correct_this = pred_result == it.answer_letter
            
#             if rationale_correct_this:
#                 rationale_correct += 1
#             if result_correct_this:
#                 result_correct += 1
#             total += 1
            
#             # Print detailed output for monitoring
#             if idx < 5:  # Print first 5 examples for debugging
#                 print(f"--- Example {idx + 1} ---")
#                 print(f"Question: {it.question[:100]}...")
#                 print(f"Expected result: {it.answer_letter}")
#                 print(f"Generated: {out_text[:100]}...")
#                 print(f"Predicted rationale: {pred_rationale[:50] if pred_rationale else 'None'}...")
#                 print(f"Predicted result: {pred_result}")
#                 print(f"Rationale correct: {rationale_correct_this}")
#                 print(f"Result correct: {result_correct_this}")
#                 print()
            
#             # Save debug outputs if requested with subcategory organization
#             if eval_debug_dir and idx < 10:  # Save first 10 examples
#                 # Create subcategory subfolder
#                 subcat_dir = os.path.join(eval_debug_dir, "samples", it.subcategory)
#                 os.makedirs(subcat_dir, exist_ok=True)
                
#                 debug_file = os.path.join(subcat_dir, f"rationale_sample_{idx:03d}.txt")
#                 with open(debug_file, "w", encoding="utf-8") as f:
#                     f.write(f"Question: {it.question}\n")
#                     f.write(f"Expected result: {it.answer_letter}\n")
#                     f.write(f"Generated: {out_text}\n")
#                     f.write(f"Predicted rationale: {pred_rationale}\n")
#                     f.write(f"Predicted result: {pred_result}\n")
#                     f.write(f"Rationale correct: {rationale_correct_this}\n")
#                     f.write(f"Result correct: {result_correct_this}\n")
#                     f.write(f"Category: {it.category}\n")
#                     f.write(f"Subcategory: {it.subcategory}\n")
#                     f.write(f"Index: {it.index}\n")
                    
#         except Exception as e:
#             print(f"Error evaluating item {idx}: {e}")
#             continue
    
#     rationale_accuracy = rationale_correct / total if total > 0 else 0.0
#     result_accuracy = result_correct / total if total > 0 else 0.0
    
#     # Save evaluation summary if debug directory is provided
#     if eval_debug_dir:
#         summary = {
#             "evaluation_type": eval_type,
#             "timestamp": datetime.now().isoformat(),
#             "rationale_accuracy": rationale_accuracy,
#             "result_accuracy": result_accuracy,
#             "total_items": total,
#             "rationale_correct": rationale_correct,
#             "result_correct": result_correct
#         }
        
#         summary_file = os.path.join(eval_debug_dir, "rationale_evaluation_summary.json")
#         with open(summary_file, "w", encoding="utf-8") as f:
#             json.dump(summary, f, indent=2, ensure_ascii=False)
#         print(f"Rationale evaluation summary saved to: {eval_debug_dir}/rationale_evaluation_summary.json")
    
#     return rationale_accuracy, result_accuracy


class ValEvalCallback(TrainerCallback):
    """
    Callback for evaluating validation accuracy during training.
    Inherits from TrainerCallback to automatically handle all callback methods.
    """
    
    def __init__(
        self, 
        processor: AutoProcessor, 
        items: List[Union[SFTItem, SFTItemAnswerOnly]], 
        device: torch.device, 
        eval_limit: Optional[int], 
        debug_dir: Optional[str], 
        system_text: str, 
        user_template: str, 
        eval_plot: bool, 
        log_file: str = None, 
        use_rationale_format: bool = False,
        train_items: List[Union[SFTItem, SFTItemAnswerOnly]] = None,
        use_consistent_samples: bool = True,
        samples_per_category: int = 3
    ):
        super().__init__()
        self.processor = processor
        self.items = items
        self.device = device
        self.eval_limit = eval_limit
        self.debug_dir = debug_dir
        self.system_text = system_text
        self.user_template = user_template
        self.eval_plot = eval_plot
        self.log_file = log_file
        self.use_rationale_format = use_rationale_format
        self.train_items = train_items
        self.use_consistent_samples = use_consistent_samples
        self.samples_per_category = samples_per_category
        self.current_train_loss = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch."""
        if model is None:
            return
        
        epoch = int(state.epoch) if state.epoch is not None else '?'
        eval_type = f"epoch_{epoch}_validation"

        val_acc = evaluate_accuracy(
            model, self.processor, self.items, self.device, 
            limit=self.eval_limit, debug_dir=self.debug_dir, 
            system_text=self.system_text, user_template=self.user_template, 
            eval_plot=self.eval_plot, use_rationale_format=self.use_rationale_format,
            use_consistent_samples=self.use_consistent_samples,
            samples_per_category=self.samples_per_category,
            eval_type=eval_type
        )
        

        # Use the captured training loss
        train_loss = self.current_train_loss
        
        # Print simplified metrics
        if train_loss is not None:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")
        else:
            print(f"[Epoch {epoch}] Val Acc: {val_acc * 100:.2f}%")
        
        # Log to file if specified
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                if train_loss is not None:
                    f.write(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc * 100:.2f}%\n")
                else:
                    f.write(f"Epoch {epoch}: Train Loss: N/A, Val Acc: {val_acc * 100:.2f}%\n")

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs:
            # Check for both 'train_loss' and 'loss' keys
            if "train_loss" in logs:
                self.current_train_loss = logs["train_loss"]
            elif "loss" in logs:
                self.current_train_loss = logs["loss"]
