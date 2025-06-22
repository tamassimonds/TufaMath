import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import numpy as np


@dataclass
class DatasetConfig:
    seq_length: int = 2048
    pad_token_id: int = 0
    eos_token_id: int = 151645


class TextDataset(Dataset):
    """
    Efficient dataset for pretraining that loads data from create_dataset.py output.
    Works with the streaming dataset format created by the create_dataset script.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        seq_length: int = 2048,
        tokenizer=None,
        config: Optional[DatasetConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig(seq_length=seq_length)
        
        # Load shard metadata
        self.shard_files = list(self.data_dir.glob("shard_*.npy"))
        self.shard_files.sort()
        
        if not self.shard_files:
            raise ValueError(f"No shard files found in {data_dir}")
        
        # Load shard info to calculate total length
        self.shard_lengths = []
        self.cumulative_lengths = [0]
        
        for shard_file in self.shard_files:
            # Load just the shape to get length
            shard_data = np.load(shard_file, mmap_mode='r')
            shard_length = len(shard_data)
            self.shard_lengths.append(shard_length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + shard_length)
        
        self.total_length = self.cumulative_lengths[-1]
        
        # Cache for loaded shards
        self.shard_cache = {}
        self.max_cache_size = 4  # Keep 4 shards in memory
        
    def __len__(self):
        return self.total_length
    
    def _get_shard_and_index(self, idx: int):
        """Get shard index and local index within shard for global index"""
        # Binary search to find shard
        left, right = 0, len(self.cumulative_lengths) - 1
        
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_lengths[mid + 1] <= idx:
                left = mid + 1
            else:
                right = mid
        
        shard_idx = left
        local_idx = idx - self.cumulative_lengths[shard_idx]
        
        return shard_idx, local_idx
    
    def _load_shard(self, shard_idx: int):
        """Load shard data with caching"""
        if shard_idx in self.shard_cache:
            return self.shard_cache[shard_idx]
        
        # Load shard
        shard_file = self.shard_files[shard_idx]
        shard_data = np.load(shard_file)
        
        # Manage cache size
        if len(self.shard_cache) >= self.max_cache_size:
            # Remove oldest shard (simple FIFO)
            oldest_key = next(iter(self.shard_cache))
            del self.shard_cache[oldest_key]
        
        self.shard_cache[shard_idx] = shard_data
        return shard_data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example"""
        shard_idx, local_idx = self._get_shard_and_index(idx)
        shard_data = self._load_shard(shard_idx)
        
        # Get sequence
        if local_idx < len(shard_data):
            sequence = shard_data[local_idx]
        else:
            # Fallback to first sequence in next shard or padding
            sequence = np.full(self.seq_length, self.config.pad_token_id, dtype=np.int64)
        
        # Ensure proper length
        if len(sequence) > self.seq_length:
            sequence = sequence[:self.seq_length]
        elif len(sequence) < self.seq_length:
            # Pad with pad_token_id
            padding = np.full(self.seq_length - len(sequence), self.config.pad_token_id, dtype=np.int64)
            sequence = np.concatenate([sequence, padding])
        
        # Convert to tensor
        input_ids = torch.from_numpy(sequence.astype(np.int64))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.config.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


class PackedDataset(Dataset):
    """
    Memory-efficient dataset that packs sequences together to minimize padding.
    Useful when you have variable-length sequences.
    """
    
    def __init__(
        self,
        sequences: List[List[int]],
        seq_length: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: int = 151645,
    ):
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Pack sequences
        self.packed_sequences = self._pack_sequences(sequences)
    
    def _pack_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Pack sequences together to minimize padding"""
        packed = []
        current_pack = []
        current_length = 0
        
        for seq in sequences:
            # Add EOS token if not present
            if seq and seq[-1] != self.eos_token_id:
                seq = seq + [self.eos_token_id]
            
            # Check if sequence fits in current pack
            if current_length + len(seq) <= self.seq_length:
                current_pack.extend(seq)
                current_length += len(seq)
            else:
                # Finalize current pack
                if current_pack:
                    # Pad to seq_length
                    while len(current_pack) < self.seq_length:
                        current_pack.append(self.pad_token_id)
                    packed.append(current_pack)
                
                # Start new pack
                if len(seq) <= self.seq_length:
                    current_pack = seq[:]
                    current_length = len(seq)
                else:
                    # Split long sequence
                    for i in range(0, len(seq), self.seq_length):
                        chunk = seq[i:i + self.seq_length]
                        while len(chunk) < self.seq_length:
                            chunk.append(self.pad_token_id)
                        packed.append(chunk)
                    current_pack = []
                    current_length = 0
        
        # Handle remaining pack
        if current_pack:
            while len(current_pack) < self.seq_length:
                current_pack.append(self.pad_token_id)
            packed.append(current_pack)
        
        return packed
    
    def __len__(self):
        return len(self.packed_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.packed_sequences[idx]
        
        input_ids = torch.tensor(sequence, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def create_data_shards(
    input_files: List[str],
    output_dir: str,
    tokenizer,
    seq_length: int = 2048,
    shard_size: int = 1000000,
    eos_token_id: int = 151645,
):
    """
    Convert text files to tokenized shards for efficient training.
    
    Args:
        input_files: List of text file paths
        output_dir: Directory to save shards
        tokenizer: Tokenizer to use
        seq_length: Maximum sequence length
        shard_size: Number of sequences per shard
        eos_token_id: EOS token ID to append
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    shard_idx = 0
    current_shard = []
    total_sequences = 0
    
    for file_path in input_files:
        print(f"Processing {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Tokenize
                tokens = tokenizer.encode(line)
                
                # Add EOS token
                if tokens and tokens[-1] != eos_token_id:
                    tokens.append(eos_token_id)
                
                # Split if too long
                if len(tokens) > seq_length:
                    for i in range(0, len(tokens), seq_length):
                        chunk = tokens[i:i + seq_length]
                        current_shard.append(chunk)
                        total_sequences += 1
                        
                        if len(current_shard) >= shard_size:
                            # Save shard
                            shard_path = output_dir / f"shard_{shard_idx:06d}.npy"
                            np.save(shard_path, current_shard)
                            print(f"Saved shard {shard_idx} with {len(current_shard)} sequences")
                            
                            current_shard = []
                            shard_idx += 1
                else:
                    current_shard.append(tokens)
                    total_sequences += 1
                    
                    if len(current_shard) >= shard_size:
                        # Save shard
                        shard_path = output_dir / f"shard_{shard_idx:06d}.npy"
                        np.save(shard_path, current_shard)
                        print(f"Saved shard {shard_idx} with {len(current_shard)} sequences")
                        
                        current_shard = []
                        shard_idx += 1
                
                if line_idx % 10000 == 0:
                    print(f"  Processed {line_idx:,} lines")
    
    # Save remaining shard
    if current_shard:
        shard_path = output_dir / f"shard_{shard_idx:06d}.npy"
        np.save(shard_path, current_shard)
        print(f"Saved final shard {shard_idx} with {len(current_shard)} sequences")
    
    # Save metadata
    metadata = {
        'total_sequences': total_sequences,
        'num_shards': shard_idx + 1 if current_shard else shard_idx,
        'seq_length': seq_length,
        'shard_size': shard_size,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {metadata['num_shards']} shards with {total_sequences:,} total sequences")