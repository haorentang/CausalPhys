#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/run_train_sft_answer_only.sh \
#     CUDA=0 \
#     DATASET_DIR=dataset/train \
#     MODEL_ID=Qwen/Qwen2-VL-7B-Instruct \
#     EPOCHS=1 BATCH_SIZE=1 LR=1e-4 ACC_STEPS=8 QUANT=4bit \
#     OUT_DIR=runs/sft_ckpt_answer_only

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1"}

# Add memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

DATASET_DIR=${DATASET_DIR:-dataset}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2-VL-7B-Instruct}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-5e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
ACC_STEPS=${ACC_STEPS:-64}
NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:-5}
QUANT=${QUANT:-bf16}
OUT_DIR=${OUT_DIR:-runs/sft_ckpt_answer_only_5e-5}
RESIZE_IMAGES=${RESIZE_IMAGES:-448x448}

python -m sft.train_sft_answer_only \
  --dataset_dir "$DATASET_DIR" \
  --model_id "$MODEL_ID" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --warmup_ratio "$WARMUP_RATIO" \
  --gradient_accum_steps "$ACC_STEPS" \
  --num_video_frames "$NUM_VIDEO_FRAMES" \
  --quant "$QUANT" \
  --output_dir "$OUT_DIR" \
  --resize_images "$RESIZE_IMAGES"\
  --debug_generation \
  --debug_tokenization