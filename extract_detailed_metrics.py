#!/usr/bin/env python3
import os
import json

def extract_metrics_from_files(statistics_path, judge_path):
    """Extract metrics from statistics.json and judge.txt files"""
    metrics = {}
    
    # Read statistics.json
    if os.path.exists(statistics_path):
        with open(statistics_path, 'r') as f:
            stats = json.load(f)
            metrics['accuracy'] = stats.get('correctness_rate', 0)
            metrics['total_questions'] = stats.get('total_questions', 0)
    else:
        metrics['accuracy'] = 0
        metrics['total_questions'] = 0
    
    # Read judge.txt
    if os.path.exists(judge_path):
        with open(judge_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ':' in line:
                    key, value = line.strip().split(': ')
                    metrics[key] = float(value)
    else:
        metrics['existence'] = 0
        metrics['parent_node'] = 0
        metrics['description'] = 0
    
    return metrics

def process_all_models():
    """Process all models and extract detailed metrics for each subcategory"""
    evaluation_results_path = '/home/tianyi/project/causalphy/evaluation_results'
    
    # Define all subcategories (16 total)
    all_subcategories = [
        'Collision_Prediction', 'deformation', 'Fluid_Flow', 'Intention_Speculation',  # Anticipation
        'behaviour_selection', 'Object_Relocation', 'tool_selection', 'trajectory',    # Goal_orientation
        'action_substitution', 'spatial_manipulation', 'temporal_shifting', 'viewpoint_transformation',  # Intervention
        'containability', 'Mechanics_Reasoning', 'optics', 'Scene_Reconstruction'      # Perception
    ]
    
    # Define metrics
    metrics_list = ['accuracy', 'existence', 'parent_node', 'description']
    
    # Get all model directories
    models = [d for d in os.listdir(evaluation_results_path) 
              if os.path.isdir(os.path.join(evaluation_results_path, d))]
    
    # Initialize the data structure: metric -> model -> subcategory
    detailed_metrics = {}
    for metric in metrics_list:
        detailed_metrics[metric] = {}
        for model in models:
            detailed_metrics[metric][model] = {}
            for subcategory in all_subcategories:
                detailed_metrics[metric][model][subcategory] = 0.0
    
    # Process each model
    for model in models:
        model_path = os.path.join(evaluation_results_path, model)
        subfolders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
        
        # Check if model has exactly 16 subfolders
        if len(subfolders) != 16:
            print(f"Skipping {model}: has {len(subfolders)} subfolders, expected 16")
            continue
        
        print(f"Processing model: {model}")
        
        # Process each subcategory
        for subcategory in all_subcategories:
            subfolder_path = os.path.join(model_path, subcategory)
            statistics_path = os.path.join(subfolder_path, 'statistics.json')
            judge_path = os.path.join(subfolder_path, 'judge.txt')
            
            metrics = extract_metrics_from_files(statistics_path, judge_path)
            
            # Store metrics in the detailed structure
            for metric in metrics_list:
                detailed_metrics[metric][model][subcategory] = metrics[metric]
    
    return detailed_metrics, models, all_subcategories, metrics_list

def write_detailed_json(detailed_metrics, models, all_subcategories, metrics_list):
    """Write detailed metrics to JSON file with metric->model->subcategory structure"""
    output_file = '/home/tianyi/project/causalphy/detailed_metrics_by_subcategory.json'
    
    with open(output_file, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    print(f"Detailed results written to {output_file}")
    return output_file

def write_metric_matrices_json(detailed_metrics, models, all_subcategories, metrics_list):
    """Write separate JSON files for each metric showing model vs subcategory matrix"""
    base_output_dir = '/home/tianyi/project/causalphy/metric_matrices'
    os.makedirs(base_output_dir, exist_ok=True)
    
    for metric in metrics_list:
        output_file = os.path.join(base_output_dir, f'{metric}_matrix.json')
        
        # Create matrix structure
        matrix_data = {}
        for model in models:
            matrix_data[model] = {}
            for subcategory in all_subcategories:
                matrix_data[model][subcategory] = detailed_metrics[metric][model][subcategory]
        
        with open(output_file, 'w') as f:
            json.dump(matrix_data, f, indent=2)
        
        print(f"Matrix for {metric} written to {output_file}")

def print_summary(detailed_metrics, models, all_subcategories, metrics_list):
    """Print summary statistics"""
    print(f"\n=== SUMMARY ===")
    print(f"Number of metrics: {len(metrics_list)}")
    print(f"Number of models: {len(models)}")
    print(f"Number of subcategories: {len(all_subcategories)}")
    print(f"Total data points: {len(metrics_list) * len(models) * len(all_subcategories)}")
    
    print(f"\nMetrics: {', '.join(metrics_list)}")
    print(f"Models: {', '.join(models)}")
    print(f"Subcategories: {', '.join(all_subcategories)}")
    
    # Print some sample data
    print(f"\n=== SAMPLE DATA ===")
    sample_metric = metrics_list[0]
    sample_model = models[0]
    sample_subcategory = all_subcategories[0]
    sample_value = detailed_metrics[sample_metric][sample_model][sample_subcategory]
    print(f"Example: {sample_metric} for {sample_model} on {sample_subcategory} = {sample_value:.4f}")

def main():
    print("Extracting detailed metrics for all models and subcategories...")
    
    # Process all models and extract detailed metrics
    detailed_metrics, models, all_subcategories, metrics_list = process_all_models()
    
    # Write detailed JSON file
    write_detailed_json(detailed_metrics, models, all_subcategories, metrics_list)
    
    # Write metric matrices as JSON
    write_metric_matrices_json(detailed_metrics, models, all_subcategories, metrics_list)
    
    # Print summary
    print_summary(detailed_metrics, models, all_subcategories, metrics_list)
    
    print(f"\nProcessing complete! Processed {len(models)} models with {len(all_subcategories)} subcategories each.")

if __name__ == "__main__":
    main()
