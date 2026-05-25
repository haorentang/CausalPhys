#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main evaluation orchestrator - DEPRECATED.

This module is deprecated. Please use the new separated system:
- generate_responses.py for response generation
- evaluate_responses.py for response evaluation  
- analyze_results.py for result analysis

This file is kept for backward compatibility only.
"""

import os
import sys
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .generate_responses import main as generate_main
from .evaluate_responses import main as evaluate_main
from .analyze_results import main as analyze_main


def main():
    """Backward compatibility wrapper for the old evaluation system."""
    
    print("⚠️  WARNING: The old evaluation system is deprecated!")
    print("📖 Please use the new separated system:")
    print("   • python evaluation/generate_responses.py - for response generation")
    print("   • python evaluation/evaluate_responses.py - for response evaluation")
    print("   • python evaluation/analyze_results.py - for result analysis")
    print("")
    print("🔄 Redirecting to new generation system...")
    print("")
    
    # Redirect to the new generation system
    generate_main()


if __name__ == "__main__":
    main()