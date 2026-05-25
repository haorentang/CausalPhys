# Evaluation Module

This module provides a modular and extensible framework for evaluating multiple vision-language models on the CausalVL dataset.

## Structure

```
evaluation/
├── __init__.py              # Module exports
├── base_evaluator.py        # Base classes and common utilities
├── model_evaluators.py      # Model registry and factory
├── qwen_evaluator.py        # QWEN-VL evaluators (API and local)
├── api_evaluators.py        # API-based model evaluators
├── generate_responses.py    # Response generation script
├── evaluate_responses.py    # Response evaluation script
├── analyze_results.py       # Results analysis script
├── test_new_system.py       # Test script for new system
├── main_evaluator.py        # Deprecated - backward compatibility
├── example_usage.py         # Usage examples
└── README.md               # This file
```

## Features

- **Modular Design**: Each model type has its own evaluator file
- **Rate Limiting**: Built-in rate limiting to prevent API quota issues
- **Judge Evaluation**: Optional LLM-based evaluation of rationale quality
- **JSON Export**: Results are automatically saved to structured JSON files
- **Progress Tracking**: Real-time progress bars during evaluation
- **Error Handling**: Robust error handling with retry logic
- **Smart Skipping**: Automatically skips subcategories that already have results
- **Incremental Processing**: Resume interrupted evaluations without losing progress
- **Parallel Processing**: Multiple models are processed in parallel for faster evaluation

## File Organization

- **`base_evaluator.py`**: Base classes and common utilities
- **`model_evaluators.py`**: Central registry and factory for all evaluators
- **`qwen_evaluator.py`**: QWEN-VL specific evaluators (API and local)
- **`api_evaluators.py`**: API-based model evaluators (GPT-4o, Gemini, LLaVA, InternVL)
- **`generate_responses.py`**: Response generation script (NEW)
- **`evaluate_responses.py`**: Response evaluation script (NEW)
- **`analyze_results.py`**: Results analysis script (NEW)
- **`main_evaluator.py`**: Deprecated - backward compatibility wrapper

## Supported Models

- **QWEN-VL-7B (API)**: `qwen-vl-7b` - Via OpenRouter API (limited to 1 image)
- **QWEN-VL-7B (Local)**: `qwen-vl-7b-local` - Local model with full multi-image support
- **GPT-4o**: `gpt-4o`
- **GPT-4o Mini**: `gpt-4o-mini` - Faster and cheaper alternative to GPT-4o
- **Gemini 2.5 Flash**: `gemini-2.5-flash`
- **LLaVA-NoE**: `llava-noe`
- **LLaVA 1.5 13B**: `llava-1.5-13b`
- **InternVL-2.5-26B**: `internvl-2.5-26b`
- **InternVL-3-78B**: `internvl-3-78b`
- **QVQ-32B**: `qvq-32b` (text-only)

## Quick Start

### Using the New Separated System

```bash
# 1. Generate responses for multiple models
python evaluation/generate_responses.py --dataset_dir ./dataset --models qwen-vl-7b-local gpt-4o-mini

# 2. Evaluate responses with judge evaluation
python evaluation/evaluate_responses.py --baseline_path evaluation_results/qwen-vl-7b-local --use_judge

# 3. Analyze and compare results
python evaluation/analyze_results.py --compare --models qwen-vl-7b-local gpt-4o-mini internvl-3-78b qvq-32b

# Generate with rate limiting (for free APIs)
python evaluation/generate_responses.py --dataset_dir ./dataset --models qwen-vl-7b --rate_limit 10

# Generate for specific subset
python evaluation/generate_responses.py --dataset_dir ./dataset --models qwen-vl-7b-local --subset optics

# Skip existing results (default behavior)
python evaluation/generate_responses.py --dataset_dir ./dataset --models qwen-vl-7b-local

# Force regeneration of all subcategories
python evaluation/generate_responses.py --dataset_dir ./dataset --models qwen-vl-7b-local --force_regenerate

# Process multiple models i

# Control parallel workers (default: 4)
python evaluation/generate_responses.py --dataset_dir ./dataset --models qwen-vl-7b-local gpt-4o-mini --max_workers 2
```

### Using the Deprecated System (Backward Compatibility)

```bash
# This will redirect to the new generation system with a warning
python evaluation/main_evaluator.py --dataset_dir /path/to/dataset --models qwen-vl-7b
```

### Using Individual Evaluators

```python
from evaluation import QWENVLEvaluator, RateLimiter
from dataset.causal_vl import CausalVLDataset

# Create evaluator
rate_limiter = RateLimiter(max_requests=10, time_window=60)
evaluator = QWENVLEvaluator(rate_limiter)

# Load dataset
dataset = CausalVLDataset("/path/to/dataset")

# Evaluate a single sample
sample = dataset.samples[0]
result = evaluator.evaluate_single_sample(
    sample, 
    "sft/prompts/sft_system.txt", 
    "sft/prompts/sft_user.txt"
)
```

### Using the Main Orchestrator

```python
from evaluation import MainEvaluator
from dataset.causal_vl import CausalVLDataset

# Create main evaluator
main_evaluator = MainEvaluator(rate_limit=10, rate_window=60)

# Load dataset
dataset = CausalVLDataset("/path/to/dataset")

# Evaluate multiple models
results = main_evaluator.evaluate_models(
    models=["qwen-vl-7b", "gpt-4o"],
    dataset=dataset,
    system_prompt_path="sft/prompts/sft_system.txt",
    user_prompt_path="sft/prompts/sft_user.txt",
    output_dir="results",
    use_judge=True
)
```

## Creating Custom Evaluators

```python
from evaluation.base_evaluator import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def __init__(self, rate_limiter=None):
        model_config = {
            "provider": "openrouter",
            "model": "my/custom-model",
            "supports_images": True,
            "max_tokens": 2048
        }
        super().__init__("my-custom-model", model_config, rate_limiter)
    
    def get_model_config(self):
        return self.model_config
```

## Configuration

### API Keys

The module supports multiple API providers. Set up your API keys using environment variables or key files:

- **OpenAI**: `OPENAI_API_KEY`
- **OpenRouter**: `OPENROUTER_API_KEY`
- **Google**: `GOOGLE_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`

### Rate Limiting

Rate limiting is **optional** and can be disabled for paid API methods:

```python
# With rate limiting (for free/limited APIs)
rate_limiter = RateLimiter(
    max_requests=10,    # Max requests per time window
    time_window=60      # Time window in seconds
)

# Without rate limiting (for paid APIs)
rate_limiter = None
```

**Command line usage:**
- Default behavior - No rate limiting (for paid APIs)
- `--rate_limit 10` - Enable rate limiting (10 requests per minute)

## Output Format

Results are saved as CSV files with the following columns:

- `sample_id`: Unique identifier for the sample
- `category`: Sample category
- `subcategory`: Sample subcategory
- `question`: The question text
- `ground_truth_answer`: Correct answer
- `model_answer`: Model's predicted answer
- `is_correct`: Whether the model was correct
- `rationale`: Model's reasoning
- `cot_steps`: Chain-of-thought steps
- `raw_response`: Full model response
- `existence_rate`: Judge evaluation metric
- `correctness_rate`: Judge evaluation metric
- `parent_match_rate`: Judge evaluation metric
- `judge_response`: Full judge response

## Examples

See `example_usage.py` for comprehensive examples of how to use the evaluation module.

## Migration from Original Script

The original `evaluate_models.py` script has been refactored into this modular structure. The new `evaluate_models_new.py` script provides the same command-line interface with improved organization and extensibility.
