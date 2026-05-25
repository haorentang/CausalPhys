#!/bin/bash
set -euo pipefail

# Script to train rationale model from a pretrained checkpoint
# Usage:
#   scripts/run_train_rationale_from_checkpoint.sh \
#     CUDA=0 \
#     PRETRAINED_MODEL_PATH=runs/sft_ckpt_answer_only_5e-5 \
#     DATASET_DIR=dataset/train \
#     EPOCHS=3 BATCH_SIZE=1 LR=2e-5 ACC_STEPS=8 QUANT=4bit \
#     OUT_DIR=runs/sft_ckpt_rationale_from_checkpoint

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1"}

# Add memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Default parameters
DATASET_DIR=${DATASET_DIR:-dataset}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-runs/sft_ckpt_answer_only_5e-5}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
ACC_STEPS=${ACC_STEPS:-64}
NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:-5}
QUANT=${QUANT:-bf16}
OUT_DIR=${OUT_DIR:-runs/sft_ckpt_rationale_from_checkpoint}
RESIZE_IMAGES=${RESIZE_IMAGES:-448x448}

# Rationale training specific parameters
RATIONALE_WEIGHT=${RATIONALE_WEIGHT:-1.0}
ANSWER_WEIGHT=${ANSWER_WEIGHT:-2.0}

echo "=========================================="
echo "Training Rationale Model from Checkpoint"
echo "=========================================="
echo "Pretrained model path: $PRETRAINED_MODEL_PATH"
echo "Dataset directory: $DATASET_DIR"
echo "Output directory: $OUT_DIR"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Rationale weight: $RATIONALE_WEIGHT"
echo "Answer weight: $ANSWER_WEIGHT"
echo "=========================================="

# Check if pretrained model path exists
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: Pretrained model path does not exist: $PRETRAINED_MODEL_PATH"
    echo "Please train an answer-only model first or provide a valid checkpoint path."
    exit 1
fi

# Check if the pretrained model has the required files
if [ ! -f "$PRETRAINED_MODEL_PATH/adapter_config.json" ]; then
    echo "Error: No adapter_config.json found in $PRETRAINED_MODEL_PATH"
    echo "This doesn't appear to be a valid LoRA checkpoint."
    exit 1
fi

echo "✅ Pretrained checkpoint found and validated"

# Run rationale training from the pretrained checkpoint
python -m sft.train_sft_rationale \
  --dataset_dir "$DATASET_DIR" \
  --model_id "$PRETRAINED_MODEL_PATH" \
  --epochs "$EPOCHS" \
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
  --debug_generation \
  --debug_tokenization

echo "=========================================="
echo "Training completed!"
echo "Final model saved to: $OUT_DIR"
echo "=========================================="
