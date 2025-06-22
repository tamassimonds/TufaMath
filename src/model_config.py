from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for Qwen2.5-1.5B architecture"""
    
    # Model architecture
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    
    # Position embeddings
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    
    # Activations and normalization
    hidden_act: str = "silu"  # SwiGLU uses SiLU
    rms_norm_eps: float = 1e-6
    
    # Attention
    attention_dropout: float = 0.0
    
    # Training settings
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True
    
    # Tokens
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: Optional[int] = None
    
    # Training optimization
    gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"
    
    def __post_init__(self):
        # Ensure head dimensions are consistent
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Set pad token to eos token if not specified
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id