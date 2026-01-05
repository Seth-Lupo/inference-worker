"""
Minimal diffusers-compatible implementations to avoid torchvision dependency.

The full diffusers package imports transformers CLIP which requires torchvision.
NVIDIA's torch 2.9.1 has ABI incompatibility with pip torchvision.
This module provides only what s3gen/matcha needs without the problematic imports.
"""
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Activations (from diffusers.models.activations)
# =============================================================================

class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, x):
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


def get_activation(act_fn: str) -> nn.Module:
    """Returns an activation function module."""
    if act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "gelu_tanh" or act_fn == "gelu-approximate":
        return nn.GELU(approximate="tanh")
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "silu" or act_fn == "swish":
        return nn.SiLU()
    elif act_fn == "mish":
        return nn.Mish()
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")


# =============================================================================
# LoRA Compatible Linear (from diffusers.models.lora)
# =============================================================================

class LoRACompatibleLinear(nn.Linear):
    """Linear layer that can optionally have LoRA adapters (we don't use them)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)


# =============================================================================
# Adaptive Layer Norm (from diffusers.models.attention)
# =============================================================================

class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm that incorporates timestep embeddings."""

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with zero initialization for scale/shift."""

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor, class_labels: torch.Tensor = None, hidden_dtype=None
    ):
        emb = self.emb(timestep)
        if class_labels is not None:
            emb = emb + class_labels
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# =============================================================================
# Attention (from diffusers.models.attention_processor)
# =============================================================================

class Attention(nn.Module):
    """Multi-head attention with optional cross-attention."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.upcast_attention = upcast_attention

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape to (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.heads, self.inner_dim // self.heads).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.inner_dim // self.heads).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.inner_dim // self.heads).transpose(1, 2)

        if self.upcast_attention:
            query = query.float()
            key = key.float()

        # Attention
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.upcast_attention:
            attn_weights = attn_weights.to(value.dtype)

        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)

        # Output projection
        for module in self.to_out:
            hidden_states = module(hidden_states)

        return hidden_states


# =============================================================================
# Decorator (from diffusers.utils.torch_utils)
# =============================================================================

def maybe_allow_in_graph(cls):
    """Decorator that marks a class as safe for torch.compile graph capture."""
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "allow_in_graph"):
        torch._dynamo.allow_in_graph(cls)
    return cls
