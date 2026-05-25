#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage of the evaluation module.

This script demonstrates how to use individual model evaluators
and the main evaluation orchestrator.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.causal_vl import CausalVLDataset
from evaluation import (
    QWENVLEvaluator, 
    GPT4oEvaluator,
    GPT4oMiniEvaluator,
    MainEvaluator,
    RateLimiter
)


def example_individual_evaluator():
    """Example of using an individual model evaluator."""
    print("=== Individual Evaluator Example ===")
    
    # Create a rate limiter
    rate_limiter = RateLimiter(max_requests=5, time_window=60)
    
    # Create a QWEN-VL evaluator
    qwen_evaluator = QWENVLEvaluator(rate_limiter)
    
    # Load a small dataset for testing
    dataset = CausalVLDataset("path/to/your/dataset")
    
    # Limit to just a few samples for testing
    dataset.samples = dataset.samples[:2]
    
    # Evaluate samples
    system_prompt = "sft/prompts/sft_system.txt"
    user_prompt = "sft/prompts/sft_user.txt"
    
    for sample in dataset.samples:
        result = qwen_evaluator.evaluate_single_sample(sample, system_prompt, user_prompt)
        print(f"Sample {result['sample_id']}: {result['model_answer']} (Correct: {result['is_correct']})")


def example_main_evaluator():
    """Example of using the main evaluation orchestrator."""
    print("\n=== Main Evaluator Example ===")
    
    # Create main evaluator
    main_evaluator = MainEvaluator(rate_limit=5, rate_window=60)
    
    # Load dataset
    dataset = CausalVLDataset("path/to/your/dataset")
    dataset.samples = dataset.samples[:5]  # Limit for testing
    
    # Evaluate specific models
    models_to_evaluate = ["qwen-vl-7b", "gpt-4o"]
    
    results = main_evaluator.evaluate_models(
        models=models_to_evaluate,
        dataset=dataset,
        system_prompt_path="sft/prompts/sft_system.txt",
        user_prompt_path="sft/prompts/sft_user.txt",
        output_dir="test_results",
        use_judge=False  # Set to True if you want judge evaluation
    )
    
    print(f"Evaluated {len(results)} models")
    for model_name, model_results in results.items():
        accuracy = sum(r["is_correct"] for r in model_results) / len(model_results)
        print(f"{model_name}: {accuracy:.3f} accuracy")


def example_gpt4o_mini():
    """Example of using GPT-4o mini evaluator."""
    print("\n=== GPT-4o Mini Evaluator Example ===")
    
    # Create GPT-4o mini evaluator (no rate limiting needed for paid API)
    evaluator = GPT4oMiniEvaluator()
    
    # Load dataset
    dataset = CausalVLDataset("dataset")
    
    # Evaluate a single sample
    sample = dataset.samples[0]
    result = evaluator.evaluate_single_sample(
        sample, 
        "sft/prompts/sft_system.txt", 
        "sft/prompts/sft_user.txt"
    )
    
    print(f"Question: {sample.question[:100]}...")
    print(f"Expected: {sample.answer_letter}")
    print(f"Predicted: {result['model_answer']}")
    print(f"Correct: {result['is_correct']}")
    print(f"Rationale: {result['rationale'][:200]}...")


def example_custom_evaluator():
    """Example of creating a custom evaluator."""
    print("\n=== Custom Evaluator Example ===")
    
    from evaluation.base_evaluator import BaseEvaluator
    
    class CustomEvaluator(BaseEvaluator):
        """Custom evaluator for a hypothetical model."""
        
        def __init__(self, rate_limiter=None):
            model_config = {
                "provider": "openrouter",
                "model": "custom/model-name",
                "supports_images": True,
                "max_tokens": 2048
            }
            super().__init__("custom-model", model_config, rate_limiter)
        
        def get_model_config(self):
            return self.model_config
        
        def evaluate_single_sample(self, sample, system_prompt_path, user_prompt_path):
            """Override with custom evaluation logic if needed."""
            # You can add custom preprocessing or postprocessing here
            result = super().evaluate_single_sample(sample, system_prompt_path, user_prompt_path)
            
            # Add custom fields
            result["custom_metric"] = "custom_value"
            
            return result
    
    # Use the custom evaluator
    custom_evaluator = CustomEvaluator()
    print(f"Custom evaluator created for model: {custom_evaluator.model_name}")


if __name__ == "__main__":
    print("Evaluation Module Examples")
    print("=" * 50)
    
    # Note: These examples require actual dataset paths and API keys
    # Uncomment and modify the paths as needed
    
    # example_individual_evaluator()
    # example_main_evaluator()
    # example_gpt4o_mini()
    # example_custom_evaluator()
    
    print("\nTo run these examples:")
    print("1. Set up your dataset path")
    print("2. Configure your API keys")
    print("3. Uncomment the example functions above")
    print("4. Run: python evaluation/example_usage.py")
