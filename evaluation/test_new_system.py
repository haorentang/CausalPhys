#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the new generation and evaluation system.
"""

import sys
import os
import json
import tempfile
import shutil

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_generation_system():
    """Test the generation system with a small sample."""
    
    print("🧪 Testing new generation and evaluation system...")
    
    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="eval_test_")
    print(f"[Info] Using test directory: {test_dir}")
    
    try:
        # Test 1: Generate responses for a small sample
        print("\n1️⃣ Testing response generation...")
        
        cmd = f"""
        python evaluation/generate_responses.py \
            --dataset_dir ./dataset \
            --models qwen-vl-7b \
            --subset trajectory \
            --max_samples 2 \
            --output_dir {test_dir} \
            --rate_limit 5
        """
        
        print(f"[Info] Running: {cmd.strip()}")
        result = os.system(cmd)
        
        if result != 0:
            print("❌ Generation test failed")
            return False
        
        # Check if files were created
        model_dir = os.path.join(test_dir, "qwen-vl-7b")
        if not os.path.exists(model_dir):
            print("❌ Model directory not created")
            return False
        
        subcategory_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if not subcategory_dirs:
            print("❌ No subcategory directories created")
            return False
        
        subcategory_dir = os.path.join(model_dir, subcategory_dirs[0])
        responses_path = os.path.join(subcategory_dir, "responses.json")
        statistics_path = os.path.join(subcategory_dir, "statistics.json")
        
        if not os.path.exists(responses_path) or not os.path.exists(statistics_path):
            print("❌ Response or statistics files not created")
            return False
        
        # Test 2: Verify file formats
        print("\n2️⃣ Testing file formats...")
        
        with open(responses_path, 'r') as f:
            responses = json.load(f)
        
        with open(statistics_path, 'r') as f:
            statistics = json.load(f)
        
        # Check responses format
        if not isinstance(responses, list) or len(responses) == 0:
            print("❌ Invalid responses format")
            return False
        
        response = responses[0]
        required_fields = ["sample_id", "annotation_path", "ground_truth_graph", 
                          "model_response_rationale", "model_final_answer", 
                          "ground_truth_answer", "is_correct"]
        
        for field in required_fields:
            if field not in response:
                print(f"❌ Missing field in responses: {field}")
                return False
        
        # Check statistics format
        required_stats = ["total_questions", "correct_answers", "wrong_answers", "correctness_rate"]
        for field in required_stats:
            if field not in statistics:
                print(f"❌ Missing field in statistics: {field}")
                return False
        
        print("✅ File formats are correct")
        
        # Test 3: Test evaluation system
        print("\n3️⃣ Testing evaluation system...")
        
        cmd = f"""
        python evaluation/evaluate_responses.py \
            --baseline_path {model_dir}
        """
        
        print(f"[Info] Running: {cmd.strip()}")
        result = os.system(cmd)
        
        if result != 0:
            print("❌ Evaluation test failed")
            return False
        
        print("✅ Evaluation system works")
        
        # Test 4: Test analysis system
        print("\n4️⃣ Testing analysis system...")
        
        cmd = f"""
        python evaluation/analyze_results.py \
            --baseline qwen-vl-7b \
            --results_dir {test_dir}
        """
        
        print(f"[Info] Running: {cmd.strip()}")
        result = os.system(cmd)
        
        if result != 0:
            print("❌ Analysis test failed")
            return False
        
        print("✅ Analysis system works")
        
        print(f"\n🎉 All tests passed! New system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test directory
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"[Info] Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    success = test_generation_system()
    sys.exit(0 if success else 1)
