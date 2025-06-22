#!/bin/bash

# Distributed training script for 8xB200 setup
# Usage: ./scripts/train_distributed.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Training parameters optimized for 8xB200
BATCH_SIZE=2048  # Global batch size - aggressive for fast training
MICRO_BATCH_SIZE=32  # Per-GPU micro batch size - standard for pretraining
SEQ_LENGTH=8192  # Long sequences for better context learning  
MAX_EPOCHS=1  # Train for 1 epoch
LEARNING_RATE=8e-4  # Scaled up for very large batch size

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
    src/train.py \
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