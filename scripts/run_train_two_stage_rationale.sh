#!/bin/bash
set -euo pipefail

# Script for two-stage training: answer-only first, then rationale
# Usage:
#   scripts/run_train_two_stage_rationale.sh \
#     CUDA=0 \
#     DATASET_DIR=dataset/train \
#     MODEL_ID=Qwen/Qwen2-VL-7B-Instruct \
#     ANSWER_ONLY_EPOCHS=5 RATIONALE_EPOCHS=3 \
#     BATCH_SIZE=1 LR=2e-5 ACC_STEPS=8 QUANT=4bit \
#     OUT_DIR=runs/sft_ckpt_two_stage_rationale

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1"}

# Add memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Default parameters
DATASET_DIR=${DATASET_DIR:-dataset}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2-VL-7B-Instruct}
ANSWER_ONLY_EPOCHS=${ANSWER_ONLY_EPOCHS:-5}
RATIONALE_EPOCHS=${RATIONALE_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
ACC_STEPS=${ACC_STEPS:-64}
NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:-5}
QUANT=${QUANT:-bf16}
OUT_DIR=${OUT_DIR:-runs/sft_ckpt_two_stage_rationale}
RESIZE_IMAGES=${RESIZE_IMAGES:-448x448}

# Rationale training specific parameters
RATIONALE_WEIGHT=${RATIONALE_WEIGHT:-1.0}
ANSWER_WEIGHT=${ANSWER_WEIGHT:-2.0}

TOTAL_EPOCHS=$((ANSWER_ONLY_EPOCHS + RATIONALE_EPOCHS))

echo "=========================================="
echo "Two-Stage Rationale Training"
echo "=========================================="
echo "Base model: $MODEL_ID"
echo "Dataset directory: $DATASET_DIR"
echo "Output directory: $OUT_DIR"
echo "Stage 1 (Answer-only): $ANSWER_ONLY_EPOCHS epochs"
echo "Stage 2 (Rationale): $RATIONALE_EPOCHS epochs"
echo "Total epochs: $TOTAL_EPOCHS"
echo "Learning rate: $LR"
echo "Rationale weight: $RATIONALE_WEIGHT"
echo "Answer weight: $ANSWER_WEIGHT"
echo "=========================================="

# Run two-stage training
python -m sft.train_sft_rationale \
  --dataset_dir "$DATASET_DIR" \
  --model_id "$MODEL_ID" \
  --epochs "$TOTAL_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --warmup_ratio "$WARMUP_RATIO" \
  --gradient_accum_steps "$ACC_STEPS" \
  --num_video_frames "$NUM_VIDEO_FRAMES" \
  --quant "$QUANT" \
  --output_dir "$OUT_DIR" \
  --resize_images "$RESIZE_IMAGES" \
  --rationale_weight "$RATIONALE_WEIGHT" \
  --answer_weight "$ANSWER_WEIGHT" \
  --enable_two_stage \
  --answer_only_epochs "$ANSWER_ONLY_EPOCHS" \
  --debug_generation \
  --debug_tokenization

echo "=========================================="
echo "Two-stage training completed!"
echo "Final model saved to: $OUT_DIR"
echo "Stage 1 checkpoint saved to: $OUT_DIR/stage1_answer_only"
echo "=========================================="
