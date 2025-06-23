"""
Dataset loader that works directly with the create_dataset.py streaming output.
This loads the exact same datasets defined in create_dataset.py CORPORA.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Union
from functools import partial

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

# Import CORPORA from create_dataset.py
import sys
sys.path.append(str(Path(__file__).parent))
from create_dataset import CORPORA, TOKENIZER_ID, dedup_iterator

logger = logging.getLogger(__name__)

def load_dataset_with_retry(hf_id, config=None, split="train", max_retries=5, base_delay=30, use_streaming=True):
    """Load dataset with exponential backoff retry for rate limits"""
    
    # First try to load non-streaming to use cache
    if use_streaming:
        try:
            logger.info(f"Attempting to load {hf_id} from cache (non-streaming)")
            if config:
                ds_cached = load_dataset(hf_id, config, split=split, trust_remote_code=True)
            else:
                ds_cached = load_dataset(hf_id, split=split, trust_remote_code=True)
            
            # Convert to streaming after loading from cache
            logger.info(f"Successfully loaded {hf_id} from cache, converting to streaming")
            return ds_cached.to_iterable_dataset()
            
        except Exception as e:
            logger.warning(f"Cache load failed for {hf_id}: {e}, falling back to streaming with retry")
    
    # Fallback to streaming with retry
    for attempt in range(max_retries):
        try:
            if config:
                ds = load_dataset(
                    hf_id, config, split=split, 
                    streaming=True, trust_remote_code=True
                )
            else:
                ds = load_dataset(
                    hf_id, split=split, 
                    streaming=True, trust_remote_code=True
                )
            
            # Test that we can access the dataset
            try:
                next(iter(ds))
                return ds
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    raise e  # Re-raise rate limit errors to trigger retry
                else:
                    logger.warning(f"Dataset {hf_id} loaded but iteration failed: {e}")
                    return ds  # Return anyway, might work later
                    
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load {hf_id} after {max_retries} attempts due to rate limiting")
                    raise e
                    
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limited loading {hf_id}, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error(f"Failed to load {hf_id}: {e}")
                raise e
    
    raise Exception(f"Failed to load {hf_id} after {max_retries} attempts")


class StreamingPretrainDataset(IterableDataset):
    """
    Streaming dataset that loads the exact same data as create_dataset.py
    but for training instead of saving to disk.
    """
    
    def __init__(
        self,
        seq_length: int = 8192,
        tokenizer_id: str = TOKENIZER_ID,
        max_tokens_per_source: Optional[Dict[str, int]] = None,
        shuffle_seed: int = 42,
    ):
        self.seq_length = seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_tokens_per_source = max_tokens_per_source or {}
        self.shuffle_seed = shuffle_seed
        
        # Initialize streaming datasets
        self._setup_streams()
    
    def _setup_streams(self):
        """Setup streaming datasets from CORPORA config"""
        print("Setting up streaming datasets...")
        
        self.streams = []
        for i, spec in enumerate(CORPORA):
            print(f"Loading {spec['hf_id']} (split: {spec['split']})...")
            
            # Load dataset with retry logic
            try:
                ds = load_dataset_with_retry(
                    spec["hf_id"],
                    config=spec.get("config"),
                    split=spec["split"]
                )
            except Exception as e:
                logger.error(f"Skipping {spec['hf_id']} due to persistent errors: {e}")
                continue
            
            # Apply filter if specified
            if spec.get("filter"):
                print(f"  Applying filter to {spec['hf_id']}")
                ds = ds.filter(spec["filter"])
            
            # Add tokenization and length calculation
            ds = ds.map(
                partial(self._tokenize_and_format, cap=spec["cap"]),
                remove_columns=ds.column_names
            )
            
            # Take only the specified cap
            ds = ds.take(spec["cap"] // 100)  # Rough approximation for token cap
            
            self.streams.append(ds)
            print(f"  Setup complete for {spec['hf_id']}")
        
        print(f"Setup {len(self.streams)} streaming datasets")
    
    def _tokenize_and_format(self, example, cap):
        """Tokenize text and format for training"""
        # Extract text content
        text = example.get("text", "")
        if not text:
            # Try other common text fields
            for field in ["content", "instruction", "response", "output"]:
                if field in example and example[field]:
                    text = example[field]
                    break
        
        if not text:
            return {"input_ids": [], "attention_mask": []}
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Add EOS token
        tokens.append(self.tokenizer.eos_token_id)
        
        # Split into chunks if too long
        chunks = []
        for i in range(0, len(tokens), self.seq_length):
            chunk = tokens[i:i + self.seq_length]
            
            # Pad if necessary
            if len(chunk) < self.seq_length:
                chunk.extend([self.tokenizer.pad_token_id] * (self.seq_length - len(chunk)))
            
            chunks.append({
                "input_ids": chunk,
                "attention_mask": [1 if token_id != self.tokenizer.pad_token_id else 0 
                                 for token_id in chunk]
            })
        
        return chunks[0] if chunks else {"input_ids": [], "attention_mask": []}
    
    def __iter__(self):
        """Iterate through the interleaved streams"""
        # Interleave all streams
        merged = interleave_datasets(
            self.streams, 
            probabilities=None,  # Equal probability
            seed=self.shuffle_seed
        )
        
        # Apply deduplication if needed
        # deduped = dedup_iterator(merged)  # Uncomment if you want deduplication
        
        for example in merged:
            if example["input_ids"]:  # Skip empty examples
                yield {
                    "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                }


def create_streaming_dataloader(
    batch_size: int = 32,
    seq_length: int = 8192,
    num_workers: int = 4,
    tokenizer_id: str = TOKENIZER_ID,
    **kwargs
):
    """Create a DataLoader with the streaming dataset"""
    from torch.utils.data import DataLoader
    
    dataset = StreamingPretrainDataset(
        seq_length=seq_length,
        tokenizer_id=tokenizer_id,
        **kwargs
    )
    
    def collate_fn(batch):
        """Collate function for batching"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the streaming dataset
    print("Testing streaming dataset...")
    
    dataset = StreamingPretrainDataset(seq_length=512)
    
    count = 0
    for batch in dataset:
        print(f"Batch {count}: input_ids shape = {batch['input_ids'].shape}")
        count += 1
        if count >= 5:  # Just test a few batches
            break
    
    print("Streaming dataset test complete!")