#!/bin/bash

# Script to run evaluation on Qwen 7B checkpoint
# Usage: ./run_evaluation.sh [checkpoint_path]

set -euo pipefail

# CUDA Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"6"}

# Configuration
CHECKPOINT_PATH=${1:-""}
DATASET_DIR=${DATASET_DIR:-"dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"evaluation_results/qwen-7b-checkpoint_vinilla"}
TEST_PERCENTAGE=${TEST_PERCENTAGE:-0.1}
SEED=${SEED:-42}

echo "=========================================="
echo "Qwen 7B Checkpoint Evaluation"
echo "=========================================="
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Dataset directory: $DATASET_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Test percentage: $TEST_PERCENTAGE (${TEST_PERCENTAGE}00%)"
echo "Random seed: $SEED"
echo "=========================================="

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory does not exist: $DATASET_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Starting evaluation..."
python evaluate_qwen_checkpoint.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_percentage "$TEST_PERCENTAGE" \
    --seed "$SEED" \
    --quant "bf16" \
    --resize_images "448x448" \
    --max_images 8

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
