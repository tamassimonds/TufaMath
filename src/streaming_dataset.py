"""
Dataset loader that works directly with the create_dataset.py streaming output.
This loads the exact same datasets defined in create_dataset.py CORPORA.
"""

import os
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
            
            # Load dataset
            ds = load_dataset(
                spec["hf_id"], 
                split=spec["split"], 
                streaming=True,
                trust_remote_code=True
            )
            
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