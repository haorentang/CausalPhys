#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import os
import json
import argparse
import time
import asyncio
from typing import Any, Dict, List, Optional, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

from openai import OpenAI
from httpx import HTTPError
from tqdm import tqdm

from dataset.causal_vl import CausalVLDataset, rationale_path_for_annotation
from evaluation.base_evaluator import load_api_key

ProviderT = Literal["openai", "openrouter", "anthropic"]


def _load_anthropic_key() -> str:
    """Read ANTHROPIC_API_KEY from env or from keys.txt (supports `VAR = "..."` with optional spaces)."""
    import re as _re
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    for p in [os.path.join(os.getcwd(), "keys.txt"), os.path.expanduser("~/.anthropic_api_key")]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
            m = _re.search(r'ANTHROPIC_API_KEY\s*=\s*"([^"]+)"', txt)
            if m:
                return m.group(1)
            m2 = _re.search(r"(sk-ant-[\w-]+)", txt)
            if m2:
                return m2.group(0)
    raise RuntimeError("ANTHROPIC_API_KEY not found in env or keys.txt")


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


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=10, time_window=60)  # 10 requests per minute


def is_free_model(model: str) -> bool:
    """Check if the model is free (no rate limiting needed)."""
    free_models = [
        "gpt-3.5-turbo",  # OpenAI free tier
        "claude-3-haiku",  # Anthropic free tier
        "llama-3.1-8b-instruct:free",  # OpenRouter free models
        "llama-3.1-70b-instruct:free",
        "qwen2.5-7b-instruct:free",
        "qwen2.5-14b-instruct:free",
        "qwen2.5-32b-instruct:free",
        "qwen2.5-72b-instruct:free",
        "gemma-2-9b-it:free",
        "gemma-2-27b-it:free",
        "mixtral-8x7b-instruct:free",
        "mixtral-8x22b-instruct:free",
        # DeepSeek free models on OpenRouter
        "deepseek-chat:free",
        "deepseek-r1:free",
    ]
    
    # Check if model name contains any free model identifier
    model_lower = model.lower()
    return any(free_model in model_lower for free_model in free_models)


def _make_client(provider: ProviderT) -> OpenAI:
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
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _convert_graph_to_name_based(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a graph from ID-based to name-based format using entities and relations."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Create a mapping from node ID to node name
    id_to_name = {}
    for node in nodes:
        node_id = node.get("id", "")
        node_name = node.get("name", "")
        if node_id and node_name:
            id_to_name[node_id] = node_name
    
    # Convert nodes to entities format (remove ID, keep name and description)
    entities = []
    for node in nodes:
        entity = {}
        if "name" in node:
            entity["name"] = node["name"]
        if "text" in node:
            entity["description"] = node["text"]
        if entity:  # Only add if it has content
            entities.append(entity)
    
    # Convert edges to relations format
    relations = []
    for edge in edges:
        from_id = edge.get("from", "")
        to_id = edge.get("to", "")
        from_name = id_to_name.get(from_id, "")
        to_name = id_to_name.get(to_id, "")
        
        if from_name and to_name:
            relations.append({
                "from": from_name,
                "to": to_name
            })
    
    return {
        "entities": entities,
        "relations": relations
    }


def build_messages(system_prompt_path: str, user_prompt_path: str, gt_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    system_tmpl = _read_text(system_prompt_path)
    user_tmpl = _read_text(user_prompt_path)

    question = gt_graph.get("question", "").strip()
    options = gt_graph.get("options", None)
    options_text = (json.dumps(options, ensure_ascii=False, indent=2) if options is not None else "null")
    gt_answer = gt_graph.get("ground_truth_answer", None)
    original_graph = gt_graph.get("graph", {})
    
    # Convert graph to name-based format
    name_based_graph = _convert_graph_to_name_based(original_graph)

    user_filled = user_tmpl.format(
        question=question,
        options=options_text,
        ground_truth_answer=(str(gt_answer) if gt_answer is not None else "null"),
        graph=json.dumps(name_based_graph, ensure_ascii=False, indent=2),
    )

    return [
        {"role": "system", "content": system_tmpl},
        {"role": "user", "content": user_filled},
    ]


def _extract_rationale(text: str) -> str:
    # For plain text output, the entire response is the rationale
    return text.strip()


def call_api(messages: List[Dict[str, Any]], model: str, provider: ProviderT, temperature: float = 0.0, max_retries: int = 5, disable_rate_limit: bool = False) -> Tuple[str, str]:
    if provider != "anthropic":
        client = _make_client(provider)
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=_load_anthropic_key())
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            # Apply rate limiting only for free models and when not disabled
            if not disable_rate_limit and is_free_model(model):
                rate_limiter.wait_if_needed()

            if provider == "anthropic":
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
                resp = client.messages.create(
                    model=model,
                    system=system_msg,
                    messages=user_msgs,
                    temperature=temperature,
                    max_tokens=1024,
                )
                raw = "".join(getattr(b, "text", "") for b in resp.content).strip()
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=400,
                )
                raw = (resp.choices[0].message.content or "").strip()
            rat = _extract_rationale(raw)
            return raw, rat
        except Exception as e:
            last_err = e
            error_str = str(e).lower()
            
            # Handle rate limit errors with longer delays
            if "rate limit" in error_str or "429" in error_str:
                if "per-min" in error_str:
                    # Rate limit per minute - wait longer
                    wait_time = min(60, 10 * (2 ** (attempt - 1)))
                    print(f"[RateLimit] Waiting {wait_time}s before retry {attempt}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    # General rate limit - exponential backoff
                    wait_time = min(30, 2 ** (attempt - 1))
                    print(f"[RateLimit] Waiting {wait_time}s before retry {attempt}/{max_retries}")
                    time.sleep(wait_time)
            else:
                # Other errors - shorter backoff
                wait_time = min(10, 1.5 ** (attempt - 1))
                time.sleep(wait_time)
    
    raise RuntimeError(f"Rationale API failed after {max_retries} attempts: {last_err!r}")


def process_single_rationale(sample, args, done_per_sub, lock) -> Optional[Tuple[str, str, str]]:
    """Process a single rationale generation task."""
    subkey = f"{sample.category}/{sample.subcategory}"
    
    # Skip if causal graph is None
    if sample.gt_graph is None:
        with lock:
            done_per_sub[subkey] = done_per_sub.get(subkey, 0) + 1
        print(f"[Skip] No causal graph for {sample.ann_path}")
        return None
    
    with lock:
        if args.max_per_subdir > 0 and done_per_sub.get(subkey, 0) >= args.max_per_subdir:
            return None
    
    out_path = rationale_path_for_annotation(sample.ann_path)
    if args.output_subdir and args.output_subdir != "rationales":
        out_path = out_path.replace("/rationales/", f"/{args.output_subdir}/")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path) and not args.overwrite:
        with lock:
            done_per_sub[subkey] = done_per_sub.get(subkey, 0) + 1
        return None

    try:
        messages = build_messages(args.system_prompt, args.user_prompt, sample.gt_graph)
        raw, rat = call_api(messages, model=args.model, provider=args.provider, disable_rate_limit=args.disable_rate_limit)
        
        with lock:
            done_per_sub[subkey] = done_per_sub.get(subkey, 0) + 1
        
        return (out_path, raw, rat)
    except Exception as e:
        print(f"[Error] Failed to process {sample.ann_path}: {e}")
        return None


def process_rationales_parallel(samples: List, args, max_workers: int = 5) -> int:
    """Process rationales in parallel using ThreadPoolExecutor."""
    done_per_sub: Dict[str, int] = {}
    lock = threading.Lock()
    num_written = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(process_single_rationale, sample, args, done_per_sub, lock): sample 
            for sample in samples
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(samples), desc="Generate rationales") as pbar:
            for future in as_completed(future_to_sample):
                result = future.result()
                pbar.update(1)
                
                if result is not None:
                    out_path, raw, rat = result
                    # Save just the rationale paragraph (plain text)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(rat + "\n")
                    num_written += 1
    
    return num_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--provider", type=str, default=os.getenv("RAT_PROVIDER", "openrouter"), choices=["openai", "openrouter", "anthropic"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-subcategory sampling")
    # Default to DeepSeek free model via OpenRouter
    parser.add_argument("--model", type=str, default=os.getenv("RAT_MODEL", "deepseek/deepseek-chat:free"))
    parser.add_argument("--system_prompt", type=str, default=os.path.join("sft", "prompts", "rationale_system.txt"))
    parser.add_argument("--user_prompt", type=str, default=os.path.join("sft", "prompts", "rationale_user.txt"))
    parser.add_argument("--max_per_subdir", type=int, default=0, help="Limit per subcategory; 0 for all")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing rationales")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing instead of parallel")
    parser.add_argument("--rate_limit", type=int, default=10, help="Max requests per minute for rate limiting (default: 10, only applies to free models)")
    parser.add_argument("--rate_window", type=int, default=60, help="Rate limit time window in seconds (default: 60)")
    parser.add_argument("--disable_rate_limit", action="store_true", help="Disable rate limiting entirely (for paid models)")
    parser.add_argument("--output_subdir", type=str, default="rationales", help="Subdirectory name (parallel to 'rationales') to write outputs into; lets you keep multiple teacher models side-by-side")
    args = parser.parse_args()

    # Configure global rate limiter based on command line args
    global rate_limiter
    rate_limiter = RateLimiter(max_requests=args.rate_limit, time_window=args.rate_window)
    
    if args.disable_rate_limit:
        print(f"[Info] Rate limiting disabled (for paid models)")
    else:
        print(f"[Info] Rate limiter configured: {args.rate_limit} requests per {args.rate_window} seconds (only applies to free models)")

    ds = CausalVLDataset(args.dataset_dir)

    if args.sequential:
        # Original sequential processing
        done_per_sub: Dict[str, int] = {}
        num_written = 0
        for s in tqdm(ds.samples, desc="Generate rationales"):
            subkey = f"{s.category}/{s.subcategory}"
            
            # Skip if causal graph is None
            if s.gt_graph is None:
                done_per_sub[subkey] = done_per_sub.get(subkey, 0) + 1
                continue
            
            if args.max_per_subdir > 0 and done_per_sub.get(subkey, 0) >= args.max_per_subdir:
                continue

            out_path = rationale_path_for_annotation(s.ann_path)
            if args.output_subdir and args.output_subdir != "rationales":
                out_path = out_path.replace("/rationales/", f"/{args.output_subdir}/")
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            if os.path.exists(out_path) and not args.overwrite:
                done_per_sub[subkey] = done_per_sub.get(subkey, 0) + 1
                continue

            messages = build_messages(args.system_prompt, args.user_prompt, s.gt_graph)
            raw, rat = call_api(messages, model=args.model, provider=args.provider, disable_rate_limit=args.disable_rate_limit)
            # Save just the rationale paragraph (plain text)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(rat + "\n")
            num_written += 1
            done_per_sub[subkey] = done_per_sub.get(subkey, 0) + 1
    else:
        # Parallel processing
        # Pre-trim samples to respect max_per_subdir so the progress bar reflects actual work
        samples_to_run = ds.samples
        if args.max_per_subdir > 0:
            import random as _random
            rng = _random.Random(args.seed)
            buckets: Dict[str, List] = {}
            for s in ds.samples:
                if s.gt_graph is None:
                    continue
                subkey = f"{s.category}/{s.subcategory}"
                buckets.setdefault(subkey, []).append(s)
            trimmed = []
            for subkey, items in buckets.items():
                rng.shuffle(items)
                trimmed.extend(items[: args.max_per_subdir])
            samples_to_run = trimmed
            print(f"[Info] Randomly sampled {len(samples_to_run)} samples ({args.max_per_subdir} per subcategory, seed={args.seed})")
        print(f"[Info] Using parallel processing with {args.max_workers} workers")
        num_written = process_rationales_parallel(samples_to_run, args, max_workers=args.max_workers)

    print(f"[Info] Wrote {num_written} rationale files.")


if __name__ == "__main__":
    main()


