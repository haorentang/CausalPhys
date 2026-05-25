#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model evaluator registry and factory.

This module provides a centralized registry of all available model evaluators
and a factory function to create evaluator instances.
"""

from typing import Dict, Any, Optional
from .base_evaluator import BaseEvaluator, RateLimiter

# Import evaluators from separate modules
from .qwen_evaluator import QWENVLEvaluator, LocalQWENVLEvaluator, LocalQWENVL3BEvaluator
from .llama_evaluator import LocalLlamaVLEvaluator
from .api_evaluators import GPT4oEvaluator, GPT4oMiniEvaluator, GeminiEvaluator, LLaVAEvaluator, InternVLEvaluator, InternVL78BEvaluator, InternVL14BEvaluator, QVQ32BEvaluator, LLaVA15Evaluator, LLaVA13BEvaluator, Pi4MultimodelEvaluator, QWENVL32BEvaluator
from .api_evaluators import Mistral32Evaluator, Claude35SonnetEvaluator


# Model registry for easy access
MODEL_EVALUATORS = {
    "qwen-vl-7b": QWENVLEvaluator,
    "qwen-vl-7b-local": LocalQWENVLEvaluator,
    "qwen-vl-3b-local": LocalQWENVL3BEvaluator,
    "gpt-4o": GPT4oEvaluator,
    "gpt-4o-mini": GPT4oMiniEvaluator,
    "gemini-2.5-flash": GeminiEvaluator,
    "llava-noe": LLaVAEvaluator,
    "llava-1.5-13b": LLaVA15Evaluator,
    "llava-13b": LLaVA13BEvaluator,
    "pi4-multimodel": Pi4MultimodelEvaluator,
    "mistral-3.2": Mistral32Evaluator,
    "claude-sonnet-4": Claude35SonnetEvaluator,
    "internvl-2.5-26b": InternVLEvaluator,
    "internvl-3-78b": InternVL78BEvaluator,
    "internvl-3-14b": InternVL14BEvaluator,
    "qwen-2.5-vl-32b": QWENVL32BEvaluator,
    "qvq-32b": QVQ32BEvaluator,
    "llama": LocalLlamaVLEvaluator,
}


def get_evaluator(model_name: str, rate_limiter=None) -> BaseEvaluator:
    """Factory function to get the appropriate evaluator for a model."""
    if model_name not in MODEL_EVALUATORS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_EVALUATORS.keys())}")
    
    evaluator_class = MODEL_EVALUATORS[model_name]
    return evaluator_class(rate_limiter)
