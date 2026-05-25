#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Response generation script for vision-language models on CausalVL dataset.

This script generates model responses and saves them in a structured JSON format
for later evaluation. The output structure is:

evaluation_results/
└── {model_name}/
    └── {subcategory}/
        ├── responses.json      # Detailed responses for each sample
        └── statistics.json     # Summary statistics for the subcategory
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.causal_vl import CausalVLDataset
from evaluation.model_evaluators import get_evaluator
from evaluation.base_evaluator import RateLimiter


def process_single_sample(sample, evaluator, system_prompt_path: str, user_prompt_path: str) -> Dict[str, Any]:
    """Process a single sample and return the response entry."""
    try:
        result = evaluator.evaluate_single_sample(sample, system_prompt_path, user_prompt_path)
        
        # Create response entry
        response_entry = {
            "sample_id": result["sample_id"],
            "annotation_path": sample.ann_path,
            "ground_truth_graph": sample.gt_graph,
            "ground_truth_rationale": sample.gt_graph.get("rationale", ""),  # If available
            "model_response_rationale": result["rationale"],
            "model_final_answer": result["model_answer"],
            "ground_truth_answer": result["ground_truth_answer"],
            "is_correct": result["is_correct"],
            "raw_response": result["raw_response"]
        }
        
        return response_entry
        
    except Exception as e:
        print(f"[Error] Failed to process sample {sample.gt_graph.get('id', 'unknown')}: {e}")
        # Add error entry
        response_entry = {
            "sample_id": sample.gt_graph.get("id", "unknown"),
            "annotation_path": sample.ann_path,
            "ground_truth_graph": sample.gt_graph,
            "ground_truth_rationale": sample.gt_graph.get("rationale", ""),
            "model_response_rationale": f"Error: {str(e)}",
            "model_final_answer": "ERROR",
            "ground_truth_answer": sample.gt_graph.get("ground_truth_answer", ""),
            "is_correct": False,
            "raw_response": ""
        }
        return response_entry


def save_responses_to_json(responses: List[Dict[str, Any]], output_dir: str, subcategory: str):
    """Save responses to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed responses
    responses_path = os.path.join(output_dir, "responses.json")
    with open(responses_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    # Calculate statistics
    total_questions = len(responses)
    correct_answers = sum(1 for r in responses if r.get("is_correct", False))
    wrong_answers = total_questions - correct_answers
    correctness_rate = correct_answers / total_questions if total_questions > 0 else 0.0
    
    statistics = {
        "subcategory": subcategory,
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "correctness_rate": correctness_rate
    }
    
    # Save statistics
    statistics_path = os.path.join(output_dir, "statistics.json")
    with open(statistics_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print(f"[Info] Saved {total_questions} responses to {responses_path}")
    print(f"[Info] Saved statistics to {statistics_path}")
    print(f"[Info] Accuracy: {correctness_rate:.3f} ({correct_answers}/{total_questions})")


def generate_responses_for_model(model_name: str, dataset: CausalVLDataset, 
                                system_prompt_path: str, user_prompt_path: str,
                                rate_limiter: RateLimiter = None,
                                max_samples: int = 0,
                                output_dir: str = "evaluation_results",
                                skip_existing: bool = True,
                                max_workers: int = 1) -> Dict[str, List[Dict[str, Any]]]:
    """Generate responses for a single model across all subcategories."""
    
    print(f"\n[Info] Generating responses for model: {model_name}")
    
    # Get the appropriate evaluator
    evaluator = get_evaluator(model_name, rate_limiter)
    
    # Group samples by subcategory
    subcategory_samples = defaultdict(list)
    for sample in dataset.samples:
        subcategory_samples[sample.subcategory].append(sample)
    
    # Limit samples if specified
    if max_samples > 0:
        for subcategory in subcategory_samples:
            subcategory_samples[subcategory] = subcategory_samples[subcategory][:max_samples]
    
    all_responses = {}
    
    # Process each subcategory
    for subcategory, samples in subcategory_samples.items():
        # Check if results already exist for this subcategory
        if skip_existing:
            baseline_dir = os.path.join(output_dir, model_name)
            subcategory_dir = os.path.join(baseline_dir, subcategory)
            responses_path = os.path.join(subcategory_dir, "responses.json")
            statistics_path = os.path.join(subcategory_dir, "statistics.json")
            
            if os.path.exists(responses_path) and os.path.exists(statistics_path):
                print(f"\n[Info] Skipping subcategory '{subcategory}' - results already exist")
                print(f"        Found: {responses_path}")
                continue
        
        print(f"\n[Info] Processing subcategory: {subcategory} ({len(samples)} samples)")
        
        responses = []
        
        # Use parallel processing if max_workers > 1
        if max_workers > 1:
            print(f"[Info] Using {max_workers} parallel workers for sample processing")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all sample processing tasks
                future_to_sample = {
                    executor.submit(process_single_sample, sample, evaluator, system_prompt_path, user_prompt_path): sample 
                    for sample in samples
                }
                # Collect results with progress bar
                with tqdm(total=len(samples), desc=f"Generating {subcategory}") as pbar:
                    for future in as_completed(future_to_sample):
                        sample = future_to_sample[future]
                        try:
                            response_entry = future.result()
                            responses.append(response_entry)
                        except Exception as e:
                            print(f"[Error] Failed to process sample {sample.gt_graph.get('id', 'unknown')}: {e}")
                            # Add error entry
                            response_entry = {
                                "sample_id": sample.gt_graph.get("id", "unknown"),
                                "annotation_path": sample.ann_path,
                                "ground_truth_graph": sample.gt_graph,
                                "ground_truth_rationale": sample.gt_graph.get("rationale", ""),
                                "model_response_rationale": f"Error: {str(e)}",
                                "model_final_answer": "ERROR",
                                "ground_truth_answer": sample.gt_graph.get("ground_truth_answer", ""),
                                "is_correct": False,
                                "raw_response": ""
                            }
                            responses.append(response_entry)
                        pbar.update(1)
        else:
            # Sequential processing (original behavior)
            for sample in tqdm(samples, desc=f"Generating {subcategory}"):
                response_entry = process_single_sample(sample, evaluator, system_prompt_path, user_prompt_path)
                responses.append(response_entry)
        
        # Save immediately after finishing this subcategory
        baseline_dir = os.path.join(output_dir, model_name)
        subcategory_dir = os.path.join(baseline_dir, subcategory)
        save_responses_to_json(responses, subcategory_dir, subcategory)

        all_responses[subcategory] = responses
    
    return all_responses


def process_single_model(model_name: str, dataset: CausalVLDataset, 
                        system_prompt_path: str, user_prompt_path: str,
                        rate_limiter: RateLimiter, max_samples: int,
                        output_dir: str, skip_existing: bool, max_workers: int = 1) -> Dict[str, Any]:
    """Process a single model and return results."""
    
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Generate responses
        all_responses = generate_responses_for_model(
            model_name, dataset, system_prompt_path, user_prompt_path,
            rate_limiter, max_samples, output_dir, skip_existing, max_workers
        )
        
        # Save responses for each subcategory
        # for subcategory, responses in all_responses.items():
        #     baseline_dir = os.path.join(output_dir, model_name)
        #     subcategory_dir = os.path.join(baseline_dir, subcategory)
            
        #     save_responses_to_json(responses, subcategory_dir, subcategory)
        
        print(f"\n✅ Completed processing model: {model_name}")
        return {"model": model_name, "status": "success", "responses": all_responses}
        
    except Exception as e:
        print(f"\n❌ Failed to process model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"model": model_name, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for CausalVL dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to evaluate")
    parser.add_argument("--system_prompt", type=str, 
                       default=os.path.join("sft", "prompts", "sft_system.txt"),
                       help="Path to system prompt file")
    parser.add_argument("--user_prompt", type=str,
                       default=os.path.join("sft", "prompts", "sft_user.txt"),
                       help="Path to user prompt file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--subset", type=str, help="Filter to specific subcategory")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of samples (0 for all)")
    parser.add_argument("--rate_limit", type=int, default=None, help="Max requests per minute (None to disable)")
    parser.add_argument("--rate_window", type=int, default=60, help="Rate limit time window in seconds")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip subcategories that already have results (default: True)")
    parser.add_argument("--force_regenerate", action="store_true", help="Force regeneration of all subcategories, even if they exist")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers for multiple models (default: 4)")
    
    args = parser.parse_args()
    
    # Create rate limiter if specified
    rate_limiter = None
    if args.rate_limit is not None:
        rate_limiter = RateLimiter(max_requests=args.rate_limit, time_window=args.rate_window)
        print(f"[Info] Rate limiter configured: {args.rate_limit} requests per {args.rate_window} seconds")
    else:
        print(f"[Info] Rate limiting disabled - using paid API methods")
    
    # Load dataset
    print(f"[Info] Loading dataset from {args.dataset_dir}")
    dataset = CausalVLDataset(args.dataset_dir)
    print(f"[Info] Loaded {len(dataset.samples)} samples")
    
    # Filter samples if subset specified
    if args.subset:
        original_count = len(dataset.samples)
        dataset.samples = [s for s in dataset.samples if args.subset.lower() in s.subcategory.lower()]
        print(f"[Info] Filtered to {len(dataset.samples)} samples in subset '{args.subset}' (from {original_count})")
    
    # Determine skip_existing behavior
    skip_existing = args.skip_existing and not args.force_regenerate
    if args.force_regenerate:
        print("[Info] Force regeneration enabled - will regenerate all subcategories")
    elif skip_existing:
        print("[Info] Will skip subcategories that already have results")
    else:
        print("[Info] Will regenerate all subcategories")
    
    # Process models in parallel if multiple models are specified
    if len(args.models) > 1:
        print(f"\n[Info] Processing {len(args.models)} models in parallel...")
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(len(args.models), args.max_workers)
        print(f"[Info] Using {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all model processing tasks
            future_to_model = {
                executor.submit(
                    process_single_model,
                    model_name, dataset, args.system_prompt, args.user_prompt,
                    rate_limiter, args.max_samples, args.output_dir, skip_existing, args.max_workers
                ): model_name for model_name in args.models
            }
            
            # Collect results as they complete
            completed_models = []
            failed_models = []
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        completed_models.append(model_name)
                    else:
                        failed_models.append(model_name)
                except Exception as e:
                    print(f"\n❌ Unexpected error processing {model_name}: {e}")
                    failed_models.append(model_name)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PARALLEL PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully completed: {len(completed_models)} models")
        for model in completed_models:
            print(f"   - {model}")
        
        if failed_models:
            print(f"❌ Failed: {len(failed_models)} models")
            for model in failed_models:
                print(f"   - {model}")
        
    else:
        # Single model - process with specified max_workers
        print(f"\n[Info] Processing single model: {args.models[0]}")
        result = process_single_model(
            args.models[0], dataset, args.system_prompt, args.user_prompt,
            rate_limiter, args.max_samples, args.output_dir, skip_existing, args.max_workers
        )
        
        if result["status"] == "success":
            print(f"\n✅ Successfully completed: {args.models[0]}")
        else:
            print(f"\n❌ Failed: {args.models[0]}")
    
    print(f"\n🎉 Response generation completed!")


if __name__ == "__main__":
    main()
