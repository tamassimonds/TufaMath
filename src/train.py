#!/usr/bin/env python3

import os
import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm

from .model import Qwen2ForCausalLM, TransformerBlock
from .model_config import ModelConfig
from .streaming_dataset import StreamingPretrainDataset


@dataclass
class TrainingConfig:
    # Model
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Training
    batch_size: int = 1024  # Global batch size reduced for memory
    micro_batch_size: int = 16  # Per-GPU micro batch size - reduced for memory
    max_epochs: int = 1
    max_steps: Optional[int] = None  # Will be calculated from epochs
    learning_rate: float = 6e-4  # Scaled for large batch size
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05  # 5% of training for warmup
    
    # Optimization
    gradient_clipping: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Data
    seq_length: int = 4096  # Optimized for H100 memory
    data_dir: str = "pretrain_dataset"
    
    # Checkpointing
    save_interval: int = 1000
    eval_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    
    # Logging
    log_interval: int = 10
    wandb_project: str = "qwen25-1.5b-pretrain"
    wandb_run_name: Optional[str] = None
    
    # System
    compile_model: bool = False  # Disabled due to FSDP + BF16 compatibility issues
    mixed_precision: bool = True
    
    def __post_init__(self):
        # Calculate gradient accumulation steps - will be dynamically set based on world_size
        # This will be recalculated in setup based on actual world_size
        pass
        

class TrainingState:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.tokens_seen = 0
        
    def state_dict(self):
        return {
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'tokens_seen': self.tokens_seen,
        }
    
    def load_state_dict(self, state_dict):
        self.step = state_dict['step']
        self.epoch = state_dict['epoch']
        self.best_loss = state_dict['best_loss']
        self.tokens_seen = state_dict['tokens_seen']


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def setup_model_and_optimizer(config: TrainingConfig, device):
    """Initialize model, optimizer, and scheduler"""
    
    # Create model
    model = Qwen2ForCausalLM(config.model_config)
    
    # Enable gradient checkpointing for memory savings
    model.model.gradient_checkpointing = True
    
    # Initialize weights
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=config.model_config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=config.model_config.initializer_range)
    
    model.apply(init_weights)
    
    # Setup FSDP for H100
    mixed_precision_policy = None
    if config.mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        return transformer_auto_wrap_policy(
            module, recurse, nonwrapped_numel, 
            transformer_layer_cls={TransformerBlock}
        )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    
    # Compile model if requested
    if config.compile_model:
        model = torch.compile(model)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    
    # Calculate training steps if using epochs
    if config.max_steps is None:
        # This will be set later when we know dataset size
        scheduler = None
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config.max_steps * config.warmup_ratio),
            num_training_steps=config.max_steps,
            num_cycles=0.5,
            last_epoch=-1,
        )
    
    return model, optimizer, scheduler


def setup_data(config: TrainingConfig, tokenizer, rank, world_size):
    """Setup training data loader using streaming dataset"""
    dataset = StreamingPretrainDataset(
        seq_length=config.seq_length,
        tokenizer_id="EleutherAI/gpt-neox-20b",  # Match create_dataset.py
        shuffle_seed=42 + rank,  # Different seed per rank
    )
    
    # For IterableDataset, we don't need DistributedSampler
    # Instead, we'll skip data based on rank
    if world_size > 1:
        # This is a simple approach - each rank will see different data
        # by skipping different amounts
        def rank_filter(item, rank_to_keep):
            # Simple hash-based filtering
            return hash(str(item)) % world_size == rank_to_keep
        
        # We'll handle this in the dataset itself or use a wrapper
        pass
    
    # Use fewer workers for streaming datasets to avoid shard warnings
    num_workers = min(1, world_size)  # 1 worker max for streaming datasets
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,  # Don't persist workers for streaming
    )
    
    return dataloader, None  # No sampler for IterableDataset


def save_checkpoint(
    model, optimizer, scheduler, training_state: TrainingState, 
    config: TrainingConfig, is_best: bool = False
):
    """Save training checkpoint"""
    if dist.get_rank() == 0:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_state': training_state.state_dict(),
            'config': config,
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{training_state.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        logging.info(f"Checkpoint saved at step {training_state.step}")


def load_checkpoint(
    model, optimizer, scheduler, training_state: TrainingState,
    checkpoint_path: str
):
    """Load training checkpoint"""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    training_state.load_state_dict(checkpoint['training_state'])
    
    logging.info(f"Resumed training from step {training_state.step}")
    return checkpoint.get('config')


def log_metrics(metrics: Dict[str, float], step: int, rank: int):
    """Log metrics to wandb and console"""
    if rank == 0:
        wandb.log(metrics, step=step)
        
        log_str = f"Step {step:,}"
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f" | {key}: {value:.4f}"
            else:
                log_str += f" | {key}: {value}"
        
        logging.info(log_str)


def train_step(model, batch, config: TrainingConfig):
    """Execute single training step"""
    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask', None)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    
    loss = outputs['loss']
    
    # Scale loss for gradient accumulation
    loss = loss / config.gradient_accumulation_steps
    
    return loss


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.cuda.current_device()
    
    # Load config and adjust for actual world size
    config = TrainingConfig()
    
    # Calculate gradient accumulation steps based on actual world size
    config.gradient_accumulation_steps = config.batch_size // (config.micro_batch_size * world_size)
    assert config.gradient_accumulation_steps >= 1, f"Batch size {config.batch_size} too small for {config.micro_batch_size} micro batch on {world_size} GPUs"
    
    if rank == 0:
        logging.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        logging.info(f"Effective batch size per GPU: {config.micro_batch_size * config.gradient_accumulation_steps}")
    
    training_state = TrainingState(config)
    
    # Initialize wandb
    if rank == 0:
        # Set wandb mode to online and disable interactive prompts
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_CONSOLE"] = "off"
        
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
            settings=wandb.Settings(console="off")
        )
    
    # Setup tokenizer - match the one used in create_dataset.py
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup model, optimizer, and scheduler
    model, optimizer, scheduler = setup_model_and_optimizer(config, device)
    
    # Setup data
    train_dataloader, train_sampler = setup_data(config, tokenizer, rank, world_size)
    
    # Setup scheduler if using epochs
    if scheduler is None and config.max_steps is None:
        # For streaming datasets, we need to estimate total steps
        # Assuming ~30B tokens total, with our batch size and sequence length
        estimated_total_tokens = 30_000_000_000  # From create_dataset.py TARGET_TOKENS
        tokens_per_step = config.batch_size * config.seq_length * world_size
        total_steps = estimated_total_tokens // tokens_per_step
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
            last_epoch=-1,
        )
        
        if rank == 0:
            logging.info(f"Estimated total steps: {total_steps:,}")
            logging.info(f"Warmup steps: {warmup_steps:,}")
            logging.info(f"Tokens per step: {tokens_per_step:,}")
    
    # Resume from checkpoint if specified
    if config.resume_from:
        load_checkpoint(model, optimizer, scheduler, training_state, config.resume_from)
    
    # Training loop
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    logging.info(f"Starting training from step {training_state.step}")
    logging.info(f"Training on {world_size} GPUs")
    logging.info(f"Effective batch size: {config.batch_size * world_size}")
    
    # For streaming datasets, calculate max steps differently
    if config.max_steps:
        max_steps = config.max_steps
    else:
        # Use the estimated total steps from scheduler setup
        estimated_total_tokens = 30_000_000_000
        tokens_per_step = config.batch_size * config.seq_length * world_size
        max_steps = estimated_total_tokens // tokens_per_step
    
    # Create progress bar for training
    if rank == 0:
        pbar = tqdm(total=max_steps, desc="Training Progress", unit="step")
        pbar.update(training_state.step)  # Update to current step if resuming
    
    while training_state.step < max_steps:
        if train_sampler:
            train_sampler.set_epoch(training_state.epoch)
        
        # Create epoch progress bar for dataloader
        dataloader_iter = iter(train_dataloader)
        if rank == 0:
            epoch_desc = f"Epoch {training_state.epoch} - Loading batches"
            dataloader_iter = tqdm(dataloader_iter, desc=epoch_desc, leave=False)
        
        for batch_idx, batch in enumerate(dataloader_iter):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Gradient accumulation loop
            optimizer.zero_grad()
            
            for micro_step in range(config.gradient_accumulation_steps):
                # Get micro batch
                start_idx = micro_step * config.micro_batch_size
                end_idx = start_idx + config.micro_batch_size
                
                micro_batch = {
                    k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward and backward pass
                loss = train_step(model, micro_batch, config)
                loss.backward()
                
                total_loss += loss.item()
            
            # Gradient clipping
            if config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            training_state.step += 1
            training_state.tokens_seen += config.batch_size * config.seq_length * world_size
            
            # Update main progress bar
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{total_loss/max(1, training_state.step % config.log_interval):.4f}",
                    'tokens': f"{training_state.tokens_seen/1e9:.2f}B"
                })
            
            # Logging
            if training_state.step % config.log_interval == 0:
                avg_loss = total_loss / config.log_interval
                current_lr = scheduler.get_last_lr()[0]
                
                # Calculate throughput
                elapsed = time.time() - start_time
                tokens_per_sec = (config.log_interval * config.batch_size * config.seq_length * world_size) / elapsed
                
                metrics = {
                    'train_loss': avg_loss,
                    'train_perplexity': math.exp(avg_loss),
                    'learning_rate': current_lr,
                    'tokens_per_second': tokens_per_sec,
                    'tokens_seen': training_state.tokens_seen,
                    'epoch': training_state.epoch,
                }
                
                log_metrics(metrics, training_state.step, rank)
                
                total_loss = 0.0
                start_time = time.time()
            
            # Save checkpoint
            if training_state.step % config.save_interval == 0:
                is_best = avg_loss < training_state.best_loss if 'avg_loss' in locals() else False
                if is_best:
                    training_state.best_loss = avg_loss
                
                save_checkpoint(
                    model, optimizer, scheduler, training_state, config, is_best
                )
            
            # Check if training is complete
            if training_state.step >= max_steps:
                break
        
        training_state.epoch += 1
    
    # Close progress bar
    if rank == 0:
        pbar.close()
    
    # Final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, training_state, config)
        wandb.finish()
    
    logging.info("Training completed!")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()