import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple

from .model_config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if position_ids is None:
        cos = cos[:q.shape[-2], :]
        sin = sin[:q.shape[-2], :]
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if num_key_value_heads < num_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, past_key_values=None):
        batch_size, seq_length = input_ids.shape
        
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=input_ids.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            attention_mask = torch.where(attention_mask == 0, float('-inf'), 0.0)
            causal_mask = causal_mask + attention_mask

        next_cache = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, past_key_value)
                    return custom_forward

                hidden_states, present_key_value = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    causal_mask,
                    position_ids,
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    past_key_value=past_key_value,
                )
            
            if use_cache:
                next_cache += (present_key_value,)

        hidden_states = self.norm(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_cache,
        }


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, use_cache=False, past_key_values=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        
        hidden_states = outputs['last_hidden_state']
        
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.get('past_key_values'),
        }


def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate non-embedding parameters
    embedding_params = model.model.embed_tokens.weight.numel()
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        embedding_params += model.lm_head.weight.numel()
    elif model.config.tie_word_embeddings:
        # If tied, don't double count
        pass
    
    non_embedding_params = total_params - embedding_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_embedding_params': non_embedding_params,
        'embedding_params': embedding_params,
    }


def format_number(num):
    """Format large numbers in human readable format"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def main():
    """Initialize model and print parameter counts"""
    from .model_config import ModelConfig
    
    config = ModelConfig()
    model = Qwen2ForCausalLM(config)
    
    param_counts = count_parameters(model)
    
    print("=" * 60)
    print("Qwen2.5-1.5B Model Parameter Summary")
    print("=" * 60)
    print(f"Total Parameters:        {format_number(param_counts['total_params']):>12} ({param_counts['total_params']:,})")
    print(f"Trainable Parameters:    {format_number(param_counts['trainable_params']):>12} ({param_counts['trainable_params']:,})")
    print(f"Non-Embedding Parameters: {format_number(param_counts['non_embedding_params']):>12} ({param_counts['non_embedding_params']:,})")
    print(f"Embedding Parameters:    {format_number(param_counts['embedding_params']):>12} ({param_counts['embedding_params']:,})")
    print("=" * 60)
    
    # Model architecture summary
    print("Model Architecture:")
    print(f"  Hidden Size:           {config.hidden_size:,}")
    print(f"  Intermediate Size:     {config.intermediate_size:,}")
    print(f"  Number of Layers:      {config.num_hidden_layers}")
    print(f"  Attention Heads:       {config.num_attention_heads}")
    print(f"  Key-Value Heads:       {config.num_key_value_heads}")
    print(f"  Vocabulary Size:       {config.vocab_size:,}")
    print(f"  Max Position Embeddings: {config.max_position_embeddings:,}")
    print(f"  Tied Word Embeddings:  {config.tie_word_embeddings}")
    print("=" * 60)


if __name__ == "__main__":
    main()