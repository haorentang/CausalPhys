#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
New comprehensive evaluation script for multiple vision-language models on CausalVL dataset.

This script uses the modular evaluation structure with separate evaluators for each model.

Usage:
    python evaluate_models_new.py --dataset_dir /path/to/dataset --models qwen-vl-7b gpt-4o
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.main_evaluator import main

if __name__ == "__main__":
    main()
