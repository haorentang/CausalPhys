#!/usr/bin/env python3
"""
Utility to count ground truth choices (A, B, C, D) across the entire dataset and per subcategory.
"""

import os
import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple


def find_all_responses_files(results_root: str) -> List[Tuple[str, str, str]]:
    """Find all responses.json files in the evaluation_results tree.
    
    Returns list of tuples: (model, subfolder, responses_path)
    """
    files: List[Tuple[str, str, str]] = []
    if not os.path.isdir(results_root):
        return files
    
    for model in sorted(os.listdir(results_root)):
        model_dir = os.path.join(results_root, model)
        if not os.path.isdir(model_dir):
            continue
        for subfolder in sorted(os.listdir(model_dir)):
            sub_dir = os.path.join(model_dir, subfolder)
            if not os.path.isdir(sub_dir):
                continue
            responses_path = os.path.join(sub_dir, "responses.json")
            if os.path.exists(responses_path):
                files.append((model, subfolder, responses_path))
    return files


def count_choices_in_file(responses_path: str) -> Tuple[Counter, Dict[str, Counter]]:
    """Count ground truth choices in a single responses.json file.
    
    Returns:
        - overall_counter: Counter of all choices across the file
        - subcat_counters: Dict mapping subcategory -> Counter of choices
    """
    overall_counter = Counter()
    subcat_counters: Dict[str, Counter] = defaultdict(Counter)
    
    try:
        with open(responses_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        
        for sample in samples:
            ground_truth = sample.get("ground_truth_graph", {}).get("ground_truth_answer", "")
            subcat = sample.get("ground_truth_graph", {}).get("sub_category", "unknown")
            
            if ground_truth in ["A", "B", "C", "D"]:
                overall_counter[ground_truth] += 1
                subcat_counters[subcat][ground_truth] += 1
            else:
                # Handle non-standard answers
                overall_counter["other"] += 1
                subcat_counters[subcat]["other"] += 1
                
    except Exception as e:
        print(f"Error reading {responses_path}: {e}")
    
    return overall_counter, dict(subcat_counters)


def print_counter_table(counter: Counter, title: str, total: int = None) -> None:
    """Print a nicely formatted table of choice counts."""
    if total is None:
        total = sum(counter.values())
    
    print(f"\n{title}")
    print("=" * len(title))
    print(f"{'Choice':<8} {'Count':<8} {'Percentage':<12}")
    print("-" * 30)
    
    for choice in ["A", "B", "C", "D", "other"]:
        count = counter.get(choice, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{choice:<8} {count:<8} {percentage:>8.2f}%")
    
    print(f"{'Total':<8} {total:<8} {100.00:>8.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Count ground truth choices across dataset")
    parser.add_argument("--results_root", default="evaluation_results", 
                       help="Root directory containing evaluation results")
    parser.add_argument("--model", default="", 
                       help="Specific model to analyze (default: all models)")
    parser.add_argument("--subfolder", default="", 
                       help="Specific subfolder to analyze (default: all subfolders)")
    parser.add_argument("--output", default="", 
                       help="Output JSON file to save results (optional)")
    
    args = parser.parse_args()
    
    # Find all responses files
    results_root = os.path.abspath(args.results_root)
    all_files = find_all_responses_files(results_root)
    
    # Filter by model/subfolder if specified
    if args.model:
        all_files = [f for f in all_files if f[0] == args.model]
    if args.subfolder:
        all_files = [f for f in all_files if f[1] == args.subfolder]
    
    if not all_files:
        print(f"No responses.json files found in {results_root}")
        if args.model:
            print(f"Model filter: {args.model}")
        if args.subfolder:
            print(f"Subfolder filter: {args.subfolder}")
        return
    
    print(f"Analyzing {len(all_files)} response files...")
    
    # Aggregate counters
    overall_counter = Counter()
    subcat_counters: Dict[str, Counter] = defaultdict(Counter)
    model_subcat_counters: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    
    # Process each file
    for model, subfolder, responses_path in all_files:
        print(f"Processing {model}/{subfolder}...")
        file_counter, file_subcat_counters = count_choices_in_file(responses_path)
        
        # Add to overall counters
        overall_counter.update(file_counter)
        for subcat, counter in file_subcat_counters.items():
            subcat_counters[subcat].update(counter)
            model_subcat_counters[(model, subfolder)].update(counter)
    
    # Print results
    print_counter_table(overall_counter, "OVERALL DATASET CHOICE DISTRIBUTION")
    
    print("\n\nCHOICE DISTRIBUTION BY SUBCATEGORY")
    print("=" * 50)
    for subcat in sorted(subcat_counters.keys()):
        total = sum(subcat_counters[subcat].values())
        print_counter_table(subcat_counters[subcat], f"Subcategory: {subcat}", total)
    
    # Print model/subfolder breakdown if multiple models
    if len(set(f[0] for f in all_files)) > 1:
        print("\n\nCHOICE DISTRIBUTION BY MODEL/SUBFOLDER")
        print("=" * 50)
        for (model, subfolder) in sorted(model_subcat_counters.keys()):
            total = sum(model_subcat_counters[(model, subfolder)].values())
            print_counter_table(model_subcat_counters[(model, subfolder)], 
                              f"Model: {model}, Subfolder: {subfolder}", total)
    
    # Save to JSON if requested
    if args.output:
        results = {
            "overall": dict(overall_counter),
            "by_subcategory": {k: dict(v) for k, v in subcat_counters.items()},
            "by_model_subfolder": {f"{k[0]}/{k[1]}": dict(v) for k, v in model_subcat_counters.items()},
            "files_analyzed": len(all_files),
            "total_samples": sum(overall_counter.values())
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
