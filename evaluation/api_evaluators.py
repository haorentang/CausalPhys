#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API-based model evaluators.

This module contains evaluators for models accessed via external APIs.
"""

from typing import Dict, Any, Optional
from .base_evaluator import BaseEvaluator, RateLimiter


class GPT4oEvaluator(BaseEvaluator):
    """Evaluator for GPT-4o model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openai", 
            "model": "gpt-4o",
            "supports_images": True,
            "max_tokens": 4096
        }
        super().__init__("gpt-4o", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class GPT4oMiniEvaluator(BaseEvaluator):
    """Evaluator for GPT-4o mini model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openai", 
            "model": "gpt-4o-mini",
            "supports_images": True,
            "max_tokens": 16384
        }
        super().__init__("gpt-4o-mini", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class GeminiEvaluator(BaseEvaluator):
    """Evaluator for Gemini 2.5 Flash model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
            "supports_images": True,
            "max_tokens": 8192
        }
        super().__init__("gemini-2.5-flash", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class LLaVAEvaluator(BaseEvaluator):
    """Evaluator for LLaVA-NoE model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "llava-hf/llava-v1.6-mistral-7b",
            "supports_images": True,
            "max_tokens": 2048
        }
        super().__init__("llava-noe", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class InternVLEvaluator(BaseEvaluator):
    """Evaluator for InternVL-2.5-26B model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "OpenGVLab/InternVL2-26B",
            "supports_images": True,
            "max_tokens": 4096
        }
        super().__init__("internvl-2.5-26b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class InternVL78BEvaluator(BaseEvaluator):
    """Evaluator for InternVL-3-78B model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "OpenGVLab/InternVL3-78B",
            "supports_images": True,
            "max_tokens": 4096
        }
        super().__init__("internvl-3-78b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class InternVL14BEvaluator(BaseEvaluator):
    """Evaluator for InternVL-3-14B model."""
    
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "OpenGVLab/InternVL3-14B",
            "supports_images": True,
            "max_tokens": 4096
        }
        super().__init__("internvl-3-14b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class QWENVL32BEvaluator(BaseEvaluator):
    """Evaluator for Qwen2.5-VL-32B Instruct model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "qwen/qwen2.5-vl-32b-instruct",
            "supports_images": True,
            "max_tokens": 4096
        }
        super().__init__("qwen-2.5-vl-32b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config

class QVQ32BEvaluator(BaseEvaluator):
    """Evaluator for QVQ-32B model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "qwen/qwq-32b",
            "supports_images": False,  # QVQ-32B doesn't support images via API
            "max_tokens": 4096
        }
        super().__init__("qvq-32b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class LLaVA15Evaluator(BaseEvaluator):
    """Evaluator for LLaVA 1.5 13B model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "llava-hf/llava-v1.5-13b",
            "supports_images": True,
            "max_tokens": 2048
        }
        super().__init__("llava-1.5-13b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class LLaVA13BEvaluator(BaseEvaluator):
    """Evaluator for LLaVA 13B model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "liuhaotian/llava-13b",
            "supports_images": True,
            "max_tokens": 2048
        }
        super().__init__("llava-13b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class Pi4MultimodelEvaluator(BaseEvaluator):
    """Evaluator for Pi4-multimodel-instruct model."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "microsoft/phi-4-multimodal-instruct",
            "supports_images": True,
            "max_tokens": 4096
        }
        super().__init__("pi4-multimodel", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class Mistral32Evaluator(BaseEvaluator):
    """Evaluator for Mistral 3.2 model (text-only)."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "mistralai/mistral-small-3.2-24b-instruct",
            "supports_images": False,
            "max_tokens": 8192
        }
        super().__init__("mistral-3.2", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class Claude35SonnetEvaluator(BaseEvaluator):
    """Evaluator for Claude Sonnet 4 model via OpenRouter."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "supports_images": True,
            "max_tokens": 4096
        }
        super().__init__("claude-sonnet-4", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


