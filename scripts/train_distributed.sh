#!/bin/bash

# Distributed training script for 8xB200 setup
# Usage: ./scripts/train_distributed.sh

set -e

# Configuration for 2xB200 test run
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Wandb configuration - set your API key here or export it beforehand
# export WANDB_API_KEY="your_api_key_here"  # Uncomment and set your key
# Or just make sure it's already set in your environment

# Training parameters optimized for 2xB200 test run
BATCH_SIZE=512  # Global batch size - reasonable for 2 GPUs
MICRO_BATCH_SIZE=32  # Per-GPU micro batch size - standard for pretraining
SEQ_LENGTH=4096  # Good context length for testing
MAX_EPOCHS=1  # Train for 1 epoch
LEARNING_RATE=3e-4  # Scaled down for smaller batch size

# Paths
DATA_DIR="pretrain_dataset"
CHECKPOINT_DIR="checkpoints"
WANDB_PROJECT="qwen25-1.5b-pretrain"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

echo "Starting distributed training on $WORLD_SIZE GPUs"
echo "Global batch size: $BATCH_SIZE"
echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "Sequence length: $SEQ_LENGTH"
echo "Max epochs: $MAX_EPOCHS"

# Launch training
torchrun \
    --standalone \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=$MASTER_PORT \
    -m src.train \
    --batch_size=$BATCH_SIZE \
    --micro_batch_size=$MICRO_BATCH_SIZE \
    --seq_length=$SEQ_LENGTH \
    --max_epochs=$MAX_EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --data_dir=$DATA_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --wandb_project=$WANDB_PROJECT \
    --compile_model \
    --mixed_precision