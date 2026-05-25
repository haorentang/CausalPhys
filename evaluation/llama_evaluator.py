#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local LLaMA-VL evaluator (based on Hugging Face Transformers).

Supports downloading from Hugging Face and local inference, with an interface aligned to BaseEvaluator.
By default, uses multimodal LLaMA models (e.g., Meta Llama 3.2-11B Vision Instruct), 
and also compatible with text-only LLaMA (automatically degrades to not supporting images).
"""

from typing import Dict, Any, Optional, List
import torch
from PIL import Image

from .base_evaluator import BaseEvaluator, RateLimiter


class LocalLlamaVLEvaluator(BaseEvaluator):
    """Local evaluator for LLaMA-VL model using transformers library."""

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_config = {
            "provider": "local",
            "model": model_name,
            "supports_images": True,
            "max_tokens": 2048,
            "device": self.device,
        }
        super().__init__("llama-vl-local", model_config, rate_limiter)

        # Load model and processor
        print(f"[Info] Loading LLaMA-VL model to {self.device} ...")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=(dtype or (torch.float16 if self.device == "cuda" else torch.float32)),
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print(f"[Info] LLaMA-VL model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[Error] Failed to load LLaMA-VL model: {e}")
            raise
    

    def _convert_messages(self, messages: List[Dict[str, Any]]):
        """将 BaseEvaluator 的 messages 转为 HF 处理器可用的格式。

        - 若支持图像：返回 (text, images)
        - 否则：返回 (text, [])
        """
        chat_messages = []
        images = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role == "system":
                chat_messages.append({"role": "system", "content": content})
            elif role == "user":
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if c.get("type") == "image_url" :
                            try:
                                import base64, io
                                data_uri = c["image_url"]["url"]
                                if data_uri.startswith("data:"):
                                    b64 = data_uri.split(",", 1)[1]
                                    image_bytes = base64.b64decode(b64)
                                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                                    images.append(image)
                                    parts.append({"type": "image", "image": image})
                                else:
                                    # 也支持文件路径/URL（尽量兼容）
                                    try:
                                        image = Image.open(data_uri).convert("RGB")
                                        images.append(image)
                                        parts.append({"type": "image", "image": image})
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        elif c.get("type") == "text":
                            parts.append({"type": "text", "text": c.get("text", "")})
                    chat_messages.append({"role": "user", "content": parts})
                else:
                    chat_messages.append({"role": "user", "content": content})

        # 对齐 Qwen 的 chat template 使用方式
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # 简单拼接回退
            system_texts = [m["content"] for m in chat_messages if m["role"] == "system"]
            user_texts = []
            for m in chat_messages:
                if m["role"] == "user":
                    c = m["content"]
                    if isinstance(c, list):
                        user_texts.append("\n".join([p.get("text", "") for p in c if p.get("type") == "text"]))
                    else:
                        user_texts.append(str(c))
            text = ("\n".join(system_texts) + "\n" + "\n".join(user_texts)).strip()

        return text, images

    def call_model_api(self, messages: list, max_retries: int = 5) -> tuple:
        """调用本地 LLaMA(VL) 模型进行推理。"""
        try:
            if self.rate_limiter is not None:
                self.rate_limiter.wait_if_needed()

            text, images = self._convert_messages(messages)

            if len(images) > 0:
                inputs = self.processor(text=[text], images=images, return_tensors="pt")
            else:
                inputs = self.processor(text=[text], return_tensors="pt")

            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config["max_tokens"],
                    do_sample=False,
                    pad_token_id=(getattr(self.processor, "tokenizer", None) or getattr(self.processor, "tokenizer_class", None)).pad_token_id if hasattr(self.processor, "tokenizer") else None,
                    eos_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, "tokenizer") else None,
                )

            prompt_len = inputs.get("input_ids").shape[1]
            full_seq_ids = generated_ids[0]
            gen_ids_only = full_seq_ids[prompt_len:]
            response = self.processor.decode(gen_ids_only, skip_special_tokens=True)

            return response.strip(), response.strip()

        except Exception as e:
            print(f"[Error] Local LLaMA inference failed: {e}")
            raise

    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


