#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base evaluator class and common utilities for model evaluation.

This module provides the foundational classes and utilities used by all model evaluators.
"""

import os
import json
import time
import threading
import base64
from typing import Any, Dict, List, Optional, Literal, Tuple
from collections import deque
from abc import ABC, abstractmethod

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from tqdm import tqdm
from PIL import Image

from ppo_vl.utils import load_api_key

ProviderT = Literal["openai", "openrouter", "google", "anthropic"]


class RateLimiter:
    """Simple rate limiter to prevent hitting API rate limits."""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit."""
        with self.lock:
            now = time.time()
            # Remove old requests outside the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # If we're at the limit, wait until the oldest request expires
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.time_window - now + 1
                if wait_time > 0:
                    print(f"[RateLimiter] Waiting {wait_time:.1f}s to avoid rate limit")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.requests and self.requests[0] <= now - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)


def _make_client(provider: ProviderT) -> OpenAI:
    """Create API client for different providers."""
    if provider == "openrouter":
        api_key = load_api_key(
            env_var="OPENROUTER_API_KEY",
            file_env_var="OPENROUTER_KEY_FILE",
            default_files=[
                os.path.expanduser("~/.openrouter_api_key"),
                os.path.join(os.getcwd(), "openai_key"),
                os.path.join(os.getcwd(), "keys.txt"),
            ],
        )
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        extra_headers = {}
        if os.getenv("OPENROUTER_HTTP_REFERER"):
            extra_headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")
        if os.getenv("OPENROUTER_X_TITLE"):
            extra_headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE")
        return OpenAI(base_url=base_url, api_key=api_key, default_headers=extra_headers or None)

    elif provider == "google":
        api_key = load_api_key(
            env_var="GOOGLE_API_KEY",
            file_env_var="GOOGLE_KEY_FILE",
            default_files=[
                os.path.expanduser("~/.google_api_key"),
                os.path.join(os.getcwd(), "google_key"),
            ],
        )
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found")
        return OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=api_key)

    elif provider == "anthropic":
        api_key = load_api_key(
            env_var="ANTHROPIC_API_KEY",
            file_env_var="ANTHROPIC_KEY_FILE",
            default_files=[
                os.path.expanduser("~/.anthropic_api_key"),
                os.path.join(os.getcwd(), "anthropic_key"),
            ],
        )
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found")
        return OpenAI(base_url="https://api.anthropic.com/v1", api_key=api_key)

    # Default to OpenAI
    api_key = load_api_key(
        env_var="OPENAI_API_KEY",
        file_env_var="OPENAI_KEY_FILE",
        default_files=[
            os.path.expanduser("~/.openai_api_key"),
            os.path.join(os.getcwd(), "openai_key"),
            os.path.join(os.getcwd(), "keys.txt"),
        ],
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


def _read_text(path: str) -> str:
    """Read text file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"[Warning] Failed to encode image {image_path}: {e}")
        return ""


def build_evaluation_messages(system_prompt_path: str, user_prompt_path: str, sample, supports_images: bool = True, model_name: str = "") -> List[Dict[str, Any]]:
    """Build messages for model evaluation using sft_system.txt format."""
    system_tmpl = _read_text(system_prompt_path)
    user_tmpl = _read_text(user_prompt_path)

    question = sample.gt_graph.get("question", "").strip()
    options = sample.gt_graph.get("options", None)
    options_text = (json.dumps(options, ensure_ascii=False, indent=2) if options is not None else "null")
    gt_answer = sample.gt_graph.get("ground_truth_answer", None)
    graph = sample.gt_graph.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    user_filled = user_tmpl.format(
        question=question,
        options=options_text,
        ground_truth_answer=(str(gt_answer) if gt_answer is not None else "null"),
        graph=json.dumps(graph, ensure_ascii=False, indent=2),
        nodes=json.dumps(nodes, ensure_ascii=False, indent=2),
        edges=json.dumps(edges, ensure_ascii=False, indent=2),
    )

    # Build user message content
    user_content = []
    
    # Add images if the model supports them and images are available
    if supports_images and hasattr(sample, 'media_paths') and sample.media_paths:
        # For QWEN-VL-7B via API, limit to 1 image to avoid API issues
        # For local QWEN-VL, allow multiple images
        max_images = 1 if "qwen" in model_name.lower() and "local" not in model_name.lower() else 12
        for i, image_path in enumerate(sample.media_paths[:max_images]):
            if os.path.exists(image_path):
                base64_image = _encode_image_to_base64(image_path)
                if base64_image:
                    # Determine image format from file extension
                    ext = os.path.splitext(image_path)[1].lower()
                    if ext in ['.jpg', '.jpeg']:
                        mime_type = "image/jpeg"
                    elif ext == '.png':
                        mime_type = "image/png"
                    else:
                        mime_type = "image/jpeg"  # Default fallback
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
    
    # Add text content
    user_content.append({
        "type": "text",
        "text": user_filled
    })

    return [
        {"role": "system", "content": system_tmpl},
        {"role": "user", "content": user_content},
    ]


def build_judge_messages(judge_prompt_path: str, question: str, options: str, gt_answer: str, 
                        cot_steps: List[str], final_answer_text: str, answer_choice: str,
                        nodes: List[Dict], edges: List[Dict], parent_map: Dict) -> List[Dict[str, Any]]:
    """Build messages for LLM judge evaluation."""
    judge_tmpl = _read_text(judge_prompt_path)
    
    filled_prompt = judge_tmpl.format(
        question=question,
        options=options,
        ground_truth_answer=gt_answer,
        cot_steps=json.dumps(cot_steps, ensure_ascii=False, indent=2),
        final_answer_text=final_answer_text,
        answer_choice=answer_choice,
        nodes=json.dumps(nodes, ensure_ascii=False, indent=2),
        edges=json.dumps(edges, ensure_ascii=False, indent=2),
        parent_map=json.dumps(parent_map, ensure_ascii=False, indent=2)
    )

    return [
        {"role": "user", "content": filled_prompt}
    ]


def extract_answer_and_rationale(response: str) -> Tuple[str, str, List[str]]:
    """Extract final answer and rationale from model response."""
    # Look for rationale tags
    rationale_match = None
    if "<rationale>" in response and "</rationale>" in response:
        start = response.find("<rationale>") + len("<rationale>")
        end = response.find("</rationale>")
        rationale_text = response[start:end].strip()
    else:
        # If no tags, use the entire response as rationale
        rationale_text = response.strip()
    
    # Extract final answer (look for single letter A, B, C, D)
    answer_match = None
    for line in reversed(response.split('\n')):
        line = line.strip()
        if len(line) == 1 and line in ['A', 'B', 'C', 'D']:
            answer_match = line
            break
    
    if not answer_match:
        # Fallback: look for any single letter in the response
        for char in reversed(response):
            if char in ['A', 'B', 'C', 'D']:
                answer_match = char
                break
    
    # Split rationale into steps (simple heuristic)
    steps = [step.strip() for step in rationale_text.split('.') if step.strip()]
    
    return answer_match or "UNKNOWN", rationale_text, steps


class BaseEvaluator(ABC):
    """Base class for all model evaluators."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any], rate_limiter: Optional[RateLimiter] = None):
        self.model_name = model_name
        self.model_config = model_config
        self.rate_limiter = rate_limiter  # Keep as None if not provided
    
    def call_model_api(self, messages: List[Dict[str, Any]], max_retries: int = 5) -> Tuple[str, str]:
        """Call model API with retry logic."""
        client = _make_client(self.model_config["provider"])
        last_err: Optional[Exception] = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # Apply rate limiting before making the request (if rate limiter is provided)
                if self.rate_limiter is not None:
                    self.rate_limiter.wait_if_needed()
                
                resp = client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.model_config["max_tokens"],
                )
                raw = (resp.choices[0].message.content or "").strip()
                return raw, raw  # For now, return raw as both raw and processed
            except Exception as e:
                last_err = e
                error_str = str(e).lower()
                
                # Handle rate limit errors with longer delays
                if "rate limit" in error_str or "429" in error_str:
                    if "per-min" in error_str:
                        wait_time = min(60, 10 * (2 ** (attempt - 1)))
                        print(f"[RateLimit] Waiting {wait_time}s before retry {attempt}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        wait_time = min(30, 2 ** (attempt - 1))
                        print(f"[RateLimit] Waiting {wait_time}s before retry {attempt}/{max_retries}")
                        time.sleep(wait_time)
                else:
                    wait_time = min(10, 1.5 ** (attempt - 1))
                    time.sleep(wait_time)
        
        raise RuntimeError(f"Model API failed after {max_retries} attempts: {last_err!r}")
    
    def evaluate_single_sample(self, sample, system_prompt_path: str, user_prompt_path: str) -> Dict[str, Any]:
        """Evaluate a single sample with the model."""
        try:
            supports_images = self.model_config.get("supports_images", False)
            messages = build_evaluation_messages(system_prompt_path, user_prompt_path, sample, supports_images, self.model_name)
            raw_response, _ = self.call_model_api(messages)
            
            answer_choice, rationale_text, cot_steps = extract_answer_and_rationale(raw_response)
            
            # Check accuracy
            gt_answer = sample.gt_graph.get("ground_truth_answer", "")
            is_correct = answer_choice == gt_answer
            
            return {
                "sample_id": sample.gt_graph.get("id", "unknown"),
                "category": sample.category,
                "subcategory": sample.subcategory,
                "question": sample.gt_graph.get("question", ""),
                "ground_truth_answer": gt_answer,
                "model_answer": answer_choice,
                "is_correct": is_correct,
                "rationale": rationale_text,
                "cot_steps": cot_steps,
                "raw_response": raw_response,
                "model_name": self.model_name
            }
        except Exception as e:
            print(f"[Error] Failed to evaluate sample {sample.gt_graph.get('id', 'unknown')} with {self.model_name}: {e}")
            return {
                "sample_id": sample.gt_graph.get("id", "unknown"),
                "category": sample.category,
                "subcategory": sample.subcategory,
                "question": sample.gt_graph.get("question", ""),
                "ground_truth_answer": sample.gt_graph.get("ground_truth_answer", ""),
                "model_answer": "ERROR",
                "is_correct": False,
                "rationale": f"Error: {str(e)}",
                "cot_steps": [],
                "raw_response": "",
                "model_name": self.model_name
            }
    
    def judge_rationale_quality(self, evaluation_result: Dict, sample, judge_model_config: Dict, 
                               judge_prompt_path: str) -> Dict[str, Any]:
        """Use LLM judge to evaluate rationale quality."""
        try:
            # Build parent map from ground truth graph
            parent_map = {}
            nodes = sample.gt_graph.get("graph", {}).get("nodes", [])
            edges = sample.gt_graph.get("graph", {}).get("edges", [])
            
            # Create node id to node mapping
            node_map = {node["id"]: node for node in nodes}
            
            # Build parent relationships
            for edge in edges:
                child_id = edge["to"]
                parent_id = edge["from"]
                if child_id not in parent_map:
                    parent_map[child_id] = []
                parent_map[child_id].append(parent_id)
            
            # Prepare judge inputs
            question = sample.gt_graph.get("question", "")
            options = json.dumps(sample.gt_graph.get("options", []), ensure_ascii=False, indent=2)
            gt_answer = str(sample.gt_graph.get("ground_truth_answer", ""))
            cot_steps = evaluation_result["cot_steps"]
            final_answer_text = evaluation_result["rationale"]
            answer_choice = evaluation_result["model_answer"]
            
            messages = build_judge_messages(
                judge_prompt_path, question, options, gt_answer,
                cot_steps, final_answer_text, answer_choice,
                nodes, edges, parent_map
            )
            
            raw_judge_response, _ = self.call_model_api(messages, judge_model_config)
            
            # Parse judge response (should be JSON)
            try:
                judge_result = json.loads(raw_judge_response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                judge_result = {
                    "items": [],
                    "summary": {
                        "existence_rate": 0.0,
                        "correctness_rate": 0.0,
                        "parent_match_rate": 0.0
                    }
                }
            
            return {
                "judge_response": raw_judge_response,
                "judge_result": judge_result,
                "existence_rate": judge_result.get("summary", {}).get("existence_rate", 0.0),
                "correctness_rate": judge_result.get("summary", {}).get("correctness_rate", 0.0),
                "parent_match_rate": judge_result.get("summary", {}).get("parent_match_rate", 0.0)
            }
        except Exception as e:
            print(f"[Error] Judge evaluation failed for sample {evaluation_result['sample_id']}: {e}")
            return {
                "judge_response": f"Error: {str(e)}",
                "judge_result": {},
                "existence_rate": 0.0,
                "correctness_rate": 0.0,
                "parent_match_rate": 0.0
            }
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Return the model configuration for this evaluator."""
        pass
