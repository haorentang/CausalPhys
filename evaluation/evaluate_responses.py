#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Response evaluation script for generated model responses.

This script evaluates previously generated responses and can perform additional
analysis such as LLM judge evaluation of rationale quality.
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.model_evaluators import get_evaluator
from evaluation.base_evaluator import RateLimiter


def load_responses(responses_path: str) -> List[Dict[str, Any]]:
    """Load responses from JSON file."""
    with open(responses_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_responses(responses: List[Dict[str, Any]], responses_path: str):
    """Save responses to JSON file."""
    with open(responses_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)


def update_statistics(responses: List[Dict[str, Any]], statistics_path: str):
    """Update statistics based on current responses."""
    total_questions = len(responses)
    correct_answers = sum(1 for r in responses if r.get("is_correct", False))
    wrong_answers = total_questions - correct_answers
    correctness_rate = correct_answers / total_questions if total_questions > 0 else 0.0
    
    statistics = {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "correctness_rate": correctness_rate
    }
    
    # Add judge evaluation statistics if available
    judge_evaluated = sum(1 for r in responses if "judge_evaluation" in r)
    if judge_evaluated > 0:
        avg_existence = sum(r.get("judge_evaluation", {}).get("existence_rate", 0) for r in responses) / judge_evaluated
        avg_correctness = sum(r.get("judge_evaluation", {}).get("correctness_rate", 0) for r in responses) / judge_evaluated
        avg_parent_match = sum(r.get("judge_evaluation", {}).get("parent_match_rate", 0) for r in responses) / judge_evaluated
        
        statistics.update({
            "judge_evaluated": judge_evaluated,
            "avg_existence_rate": avg_existence,
            "avg_correctness_rate": avg_correctness,
            "avg_parent_match_rate": avg_parent_match
        })
    
    with open(statistics_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)


def run_judge_evaluation(responses: List[Dict[str, Any]], judge_model: str, 
                        judge_prompt_path: str, rate_limiter: RateLimiter = None) -> List[Dict[str, Any]]:
    """Run LLM judge evaluation on responses."""
    
    print(f"[Info] Running judge evaluation with model: {judge_model}")
    
    # Create judge evaluator
    judge_evaluator = get_evaluator(judge_model, rate_limiter)
    
    # Configure judge model
    judge_model_config = {
        "provider": "openrouter",
        "model": judge_model,
        "supports_images": False,
        "max_tokens": 4096
    }
    
    updated_responses = []
    
    for response in tqdm(responses, desc="Judge evaluation"):
        try:
            # Create a mock sample object for judge evaluation
            class MockSample:
                def __init__(self, response_data):
                    self.gt_graph = response_data["ground_truth_graph"]
                    self.category = "unknown"
                    self.subcategory = "unknown"
            
            mock_sample = MockSample(response)
            
            # Create evaluation result for judge
            evaluation_result = {
                "sample_id": response["sample_id"],
                "rationale": response["model_response_rationale"],
                "model_answer": response["model_final_answer"],
                "cot_steps": response["model_response_rationale"].split('.') if response["model_response_rationale"] else []
            }
            
            # Run judge evaluation
            judge_result = judge_evaluator.judge_rationale_quality(
                evaluation_result, mock_sample, judge_model_config, judge_prompt_path
            )
            
            # Add judge evaluation to response
            response["judge_evaluation"] = {
                "existence_rate": judge_result["existence_rate"],
                "correctness_rate": judge_result["correctness_rate"],
                "parent_match_rate": judge_result["parent_match_rate"],
                "judge_response": judge_result["judge_response"]
            }
            
        except Exception as e:
            print(f"[Warning] Judge evaluation failed for sample {response['sample_id']}: {e}")
            response["judge_evaluation"] = {
                "existence_rate": 0.0,
                "correctness_rate": 0.0,
                "parent_match_rate": 0.0,
                "judge_response": f"Error: {str(e)}"
            }
        
        updated_responses.append(response)
    
    return updated_responses


def evaluate_responses_for_baseline(baseline_path: str, judge_model: str = None, 
                                   judge_prompt_path: str = None, rate_limiter: RateLimiter = None):
    """Evaluate responses for a single baseline."""
    
    print(f"\n[Info] Evaluating responses in: {baseline_path}")
    
    # Find all subcategory directories
    subcategory_dirs = []
    for item in os.listdir(baseline_path):
        item_path = os.path.join(baseline_path, item)
        if os.path.isdir(item_path):
            responses_path = os.path.join(item_path, "responses.json")
            statistics_path = os.path.join(item_path, "statistics.json")
            if os.path.exists(responses_path):
                subcategory_dirs.append((item, item_path, responses_path, statistics_path))
    
    if not subcategory_dirs:
        print(f"[Warning] No subcategory directories found in {baseline_path}")
        return
    
    # Process each subcategory
    for subcategory, subcategory_path, responses_path, statistics_path in subcategory_dirs:
        print(f"\n[Info] Processing subcategory: {subcategory}")
        
        try:
            # Load responses
            responses = load_responses(responses_path)
            print(f"[Info] Loaded {len(responses)} responses")
            
            # Run judge evaluation if requested
            if judge_model and judge_prompt_path:
                responses = run_judge_evaluation(responses, judge_model, judge_prompt_path, rate_limiter)
                # Save updated responses
                save_responses(responses, responses_path)
            
            # Update statistics
            update_statistics(responses, statistics_path)
            
            # Print summary
            correct = sum(1 for r in responses if r.get("is_correct", False))
            total = len(responses)
            accuracy = correct / total if total > 0 else 0.0
            print(f"[Info] {subcategory}: {accuracy:.3f} accuracy ({correct}/{total})")
            
        except Exception as e:
            print(f"[Error] Failed to process subcategory {subcategory}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated model responses")
    parser.add_argument("--baseline_path", type=str, required=True, 
                       help="Path to baseline directory (e.g., evaluation_results/model_name)")
    parser.add_argument("--judge_model", type=str, default="deepseek/deepseek-chat-v3.1:free",
                       help="Model to use as judge for rationale evaluation")
    parser.add_argument("--judge_prompt", type=str,
                       default=os.path.join("prompts", "judge_prompt.txt"),
                       help="Path to judge prompt file")
    parser.add_argument("--use_judge", action="store_true", 
                       help="Enable LLM judge evaluation of rationale quality")
    parser.add_argument("--rate_limit", type=int, default=None, 
                       help="Max requests per minute for judge model (None to disable)")
    parser.add_argument("--rate_window", type=int, default=60, 
                       help="Rate limit time window in seconds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline_path):
        print(f"[Error] Baseline path does not exist: {args.baseline_path}")
        return
    
    # Create rate limiter if specified
    rate_limiter = None
    if args.rate_limit is not None:
        rate_limiter = RateLimiter(max_requests=args.rate_limit, time_window=args.rate_window)
        print(f"[Info] Rate limiter configured: {args.rate_limit} requests per {args.rate_window} seconds")
    else:
        print(f"[Info] Rate limiting disabled - using paid API methods")
    
    # Check if judge evaluation is requested
    if args.use_judge:
        if not os.path.exists(args.judge_prompt):
            print(f"[Error] Judge prompt file not found: {args.judge_prompt}")
            return
        print(f"[Info] Judge evaluation enabled with model: {args.judge_model}")
    else:
        args.judge_model = None
        args.judge_prompt = None
    
    # Evaluate responses
    evaluate_responses_for_baseline(
        args.baseline_path, 
        args.judge_model, 
        args.judge_prompt, 
        rate_limiter
    )
    
    print(f"\n🎉 Response evaluation completed!")


if __name__ == "__main__":
    main()
