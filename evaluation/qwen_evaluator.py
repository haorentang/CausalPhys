#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QWEN-VL model evaluators.

This module contains evaluators for QWEN-VL models, both API-based and local.
"""

from typing import Dict, Any, Optional
import torch
from PIL import Image
from .base_evaluator import BaseEvaluator, RateLimiter


class QWENVLEvaluator(BaseEvaluator):
    """Evaluator for QWEN-VL-7B model via OpenRouter API."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        model_config = {
            "provider": "openrouter",
            "model": "qwen/qwen-2.5-vl-7b-instruct",
            "supports_images": True,
            "max_tokens": 2048
        }
        super().__init__("qwen-vl-7b", model_config, rate_limiter)
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


class LocalQWENVLEvaluator(BaseEvaluator):
    """Local evaluator for QWEN-VL-7B model using transformers library."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_config = {
            "provider": "local",
            "model": model_name,
            "supports_images": True,
            "max_tokens": 2048,
            "device": self.device
        }
        super().__init__("qwen-vl-7b-local", model_config, rate_limiter)
        
        # Load model and processor
        print(f"[Info] Loading QWEN-VL model locally on {self.device}...")
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            print(f"[Info] QWEN-VL model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[Error] Failed to load QWEN-VL model: {e}")
            raise
    
    def call_model_api(self, messages: list, max_retries: int = 5) -> tuple:
        """Call the local QWEN-VL model."""
        try:
            # Apply rate limiting if provided
            if self.rate_limiter is not None:
                self.rate_limiter.wait_if_needed()
            
            # Convert messages to QWEN-VL format (following eval_utils.py approach)
            qwen_messages = []
            images = []
            
            for message in messages:
                if message["role"] == "system":
                    qwen_messages.append({
                        "role": "system",
                        "content": message["content"]
                    })
                elif message["role"] == "user" and isinstance(message["content"], list):
                    content_list = []
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            # Decode base64 image
                            import base64
                            import io
                            
                            image_data = content["image_url"]["url"].split(",")[1]
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            images.append(image)
                            
                            content_list.append({"type": "image", "image": image})
                        elif content["type"] == "text":
                            content_list.append({
                                "type": "text",
                                "text": content["text"]
                            })
                    
                    qwen_messages.append({
                        "role": "user",
                        "content": content_list
                    })
            
            # Apply chat template to get properly formatted text with image tokens
            text = self.processor.apply_chat_template(
                qwen_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Prepare inputs using the same approach as eval_utils.py
            inputs = self.processor(
                text=[text], 
                images=images, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config["max_tokens"],
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Extract generated text (following eval_utils.py approach)
            prompt_len = inputs["input_ids"].shape[1]
            full_seq_ids = generated_ids[0]
            gen_ids_only = full_seq_ids[prompt_len:]
            response = self.processor.decode(gen_ids_only, skip_special_tokens=True)
            
            return response.strip(), response.strip()
            
        except Exception as e:
            print(f"[Error] Local QWEN-VL inference failed: {e}")
            raise
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config




class LocalQWENVL3BEvaluator(BaseEvaluator):
    """Local evaluator for QWEN-VL-3B model using transformers library."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_config = {
            "provider": "local",
            "model": model_name,
            "supports_images": True,
            "max_tokens": 2048,
            "device": self.device
        }
        super().__init__("qwen-vl-3b-local", model_config, rate_limiter)
        
        # Load model and processor
        print(f"[Info] Loading QWEN-VL-3B model locally on {self.device}...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            print(f"[Info] QWEN-VL-3B model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[Error] Failed to load QWEN-VL-3B model: {e}")
            raise
    
    def call_model_api(self, messages: list, max_retries: int = 5) -> tuple:
        """Call the local QWEN-VL-3B model."""
        try:
            # Apply rate limiting if provided
            if self.rate_limiter is not None:
                self.rate_limiter.wait_if_needed()
            
            # Convert messages to QWEN-VL format (following eval_utils.py approach)
            qwen_messages = []
            images = []
            
            for message in messages:
                if message["role"] == "system":
                    qwen_messages.append({
                        "role": "system",
                        "content": message["content"]
                    })
                elif message["role"] == "user" and isinstance(message["content"], list):
                    content_list = []
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            # Decode base64 image
                            import base64
                            import io
                            
                            image_data = content["image_url"]["url"].split(",")[1]
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            images.append(image)
                            
                            content_list.append({"type": "image", "image": image})
                        elif content["type"] == "text":
                            content_list.append({
                                "type": "text",
                                "text": content["text"]
                            })
                    
                    qwen_messages.append({
                        "role": "user",
                        "content": content_list
                    })
            
            # Apply chat template to get properly formatted text with image tokens
            text = self.processor.apply_chat_template(
                qwen_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Prepare inputs using the same approach as eval_utils.py
            inputs = self.processor(
                text=[text], 
                images=images, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config["max_tokens"],
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Extract generated text (following eval_utils.py approach)
            prompt_len = inputs["input_ids"].shape[1]
            full_seq_ids = generated_ids[0]
            gen_ids_only = full_seq_ids[prompt_len:]
            response = self.processor.decode(gen_ids_only, skip_special_tokens=True)
            
            return response.strip(), response.strip()
            
        except Exception as e:
            print(f"[Error] Local QWEN-VL-3B inference failed: {e}")
            raise
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config
