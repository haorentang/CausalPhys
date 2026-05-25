#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/run_generate_rationales.sh \
#     DATASET_DIR=dataset \
#     PROVIDER=openrouter \
#     MODEL=deepseek/deepseek-chat:free \
#     MAX_PER_SUBDIR=0 \
#     OVERWRITE=false \
#     DISABLE_RATE_LIMIT=false \
#     RATE_LIMIT=10 \
#     RATE_WINDOW=60

DATASET_DIR=${DATASET_DIR:-dataset}
PROVIDER=${PROVIDER:-openrouter}
MODEL=${MODEL:-OpenGVLab/InternVL3-78B}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-sft/prompts/rationale_system.txt}
USER_PROMPT=${USER_PROMPT:-sft/prompts/rationale_user.txt}
MAX_PER_SUBDIR=${MAX_PER_SUBDIR:-0}
OVERWRITE=${OVERWRITE:-false}
DISABLE_RATE_LIMIT=${DISABLE_RATE_LIMIT:-false}
RATE_LIMIT=${RATE_LIMIT:-10}
RATE_WINDOW=${RATE_WINDOW:-60}

OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "true" ]; then
  OVERWRITE_FLAG="--overwrite"
fi

DISABLE_RATE_LIMIT_FLAG=""
if [ "$DISABLE_RATE_LIMIT" = "true" ]; then
  DISABLE_RATE_LIMIT_FLAG="--disable_rate_limit"
fi

python -m sft.generate_rationales \
  --dataset_dir "$DATASET_DIR" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --system_prompt "$SYSTEM_PROMPT" \
  --user_prompt "$USER_PROMPT" \
  --max_per_subdir "$MAX_PER_SUBDIR" \
  --max_workers 5 \
  --rate_limit "$RATE_LIMIT" \
  --rate_window "$RATE_WINDOW" \
  $OVERWRITE_FLAG \
  $DISABLE_RATE_LIMIT_FLAG


