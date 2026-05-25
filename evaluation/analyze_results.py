#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Results analysis script for generated model responses.

This script provides various analysis capabilities for the generated responses,
including cross-model comparisons and detailed breakdowns.
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, List
from collections import defaultdict
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_statistics(baseline_path: str) -> Dict[str, Dict[str, Any]]:
    """Load statistics from all subcategories in a baseline."""
    statistics = {}
    
    for item in os.listdir(baseline_path):
        item_path = os.path.join(baseline_path, item)
        if os.path.isdir(item_path):
            statistics_path = os.path.join(item_path, "statistics.json")
            if os.path.exists(statistics_path):
                with open(statistics_path, 'r', encoding='utf-8') as f:
                    statistics[item] = json.load(f)
    
    return statistics


def compare_models(results_dir: str, models: List[str] = None) -> pd.DataFrame:
    """Compare results across multiple models."""
    
    if models is None:
        # Find all model directories
        models = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d))]
    
    comparison_data = []
    
    for model in models:
        model_path = os.path.join(results_dir, model)
        if not os.path.exists(model_path):
            continue
        
        statistics = load_statistics(model_path)
        
        for subcategory, stats in statistics.items():
            comparison_data.append({
                "model": model,
                "subcategory": subcategory,
                "total_questions": stats["total_questions"],
                "correct_answers": stats["correct_answers"],
                "wrong_answers": stats["wrong_answers"],
                "correctness_rate": stats["correctness_rate"],
                "avg_existence_rate": stats.get("avg_existence_rate", None),
                "avg_correctness_rate": stats.get("avg_correctness_rate", None),
                "avg_parent_match_rate": stats.get("avg_parent_match_rate", None)
            })
    
    return pd.DataFrame(comparison_data)


def analyze_error_patterns(baseline_path: str) -> Dict[str, Any]:
    """Analyze error patterns in responses."""
    
    error_analysis = {
        "total_samples": 0,
        "error_samples": 0,
        "error_types": defaultdict(int),
        "subcategory_errors": defaultdict(int)
    }
    
    for item in os.listdir(baseline_path):
        item_path = os.path.join(baseline_path, item)
        if os.path.isdir(item_path):
            responses_path = os.path.join(item_path, "responses.json")
            if os.path.exists(responses_path):
                with open(responses_path, 'r', encoding='utf-8') as f:
                    responses = json.load(f)
                
                for response in responses:
                    error_analysis["total_samples"] += 1
                    
                    if not response.get("is_correct", False):
                        error_analysis["error_samples"] += 1
                        error_analysis["subcategory_errors"][item] += 1
                        
                        # Analyze error types
                        model_answer = response.get("model_final_answer", "")
                        if model_answer == "ERROR":
                            error_analysis["error_types"]["API_ERROR"] += 1
                        elif model_answer == "UNKNOWN":
                            error_analysis["error_types"]["NO_ANSWER"] += 1
                        else:
                            error_analysis["error_types"]["WRONG_ANSWER"] += 1
    
    return error_analysis


def print_summary_table(df: pd.DataFrame):
    """Print a formatted summary table."""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Overall accuracy by model
    model_summary = df.groupby("model").agg({
        "total_questions": "sum",
        "correct_answers": "sum",
        "correctness_rate": "mean"
    }).round(3)
    
    print("\nOverall Performance:")
    print("-" * 50)
    for model, row in model_summary.iterrows():
        print(f"{model:20} | {row['correctness_rate']:.3f} | {row['correct_answers']:4d}/{row['total_questions']:4d}")
    
    # Performance by subcategory
    print("\nPerformance by Subcategory:")
    print("-" * 50)
    subcategory_summary = df.groupby("subcategory").agg({
        "correctness_rate": ["mean", "std", "count"]
    }).round(3)
    
    for subcategory, row in subcategory_summary.iterrows():
        mean_acc = row[("correctness_rate", "mean")]
        std_acc = row[("correctness_rate", "std")]
        count = row[("correctness_rate", "count")]
        print(f"{subcategory:20} | {mean_acc:.3f} ± {std_acc:.3f} | {count} models")


def main():
    parser = argparse.ArgumentParser(description="Analyze generated model responses")
    parser.add_argument("--results_dir", type=str, default="evaluation_results",
                       help="Directory containing all baseline results")
    parser.add_argument("--baseline", type=str, help="Specific baseline to analyze")
    parser.add_argument("--models", nargs="+", help="Specific models to compare")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--error_analysis", action="store_true", help="Analyze error patterns")
    parser.add_argument("--output_csv", type=str, help="Save comparison to CSV file")
    
    args = parser.parse_args()
    
    if args.baseline:
        # Analyze single baseline
        baseline_path = os.path.join(args.results_dir, args.baseline)
        if not os.path.exists(baseline_path):
            print(f"[Error] Baseline path does not exist: {baseline_path}")
            return
        
        print(f"[Info] Analyzing baseline: {args.baseline}")
        
        if args.error_analysis:
            error_analysis = analyze_error_patterns(baseline_path)
            print(f"\nError Analysis for {args.baseline}:")
            print(f"Total samples: {error_analysis['total_samples']}")
            print(f"Error samples: {error_analysis['error_samples']}")
            print(f"Error rate: {error_analysis['error_samples']/error_analysis['total_samples']:.3f}")
            print(f"\nError types: {dict(error_analysis['error_types'])}")
            print(f"Errors by subcategory: {dict(error_analysis['subcategory_errors'])}")
        
        # Load and display statistics
        statistics = load_statistics(baseline_path)
        print(f"\nStatistics for {args.baseline}:")
        for subcategory, stats in statistics.items():
            print(f"  {subcategory}: {stats['correctness_rate']:.3f} ({stats['correct_answers']}/{stats['total_questions']})")
    
    elif args.compare:
        # Compare multiple models
        print(f"[Info] Comparing models in: {args.results_dir}")
        
        df = compare_models(args.results_dir, args.models)
        if df.empty:
            print("[Warning] No data found for comparison")
            return
        
        print_summary_table(df)
        
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"\n[Info] Comparison saved to: {args.output_csv}")
    
    else:
        print("[Error] Please specify --baseline or --compare")
        return


if __name__ == "__main__":
    main()
