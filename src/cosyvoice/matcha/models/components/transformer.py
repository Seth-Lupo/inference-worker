"""BasicTransformerBlock from Matcha-TTS - inlined to avoid dependency."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import maybe_allow_in_graph

# Handle different diffusers versions
try:
    from diffusers.models.lora import LoRACompatibleLinear
except ImportError:
    LoRACompatibleLinear = nn.Linear

# Handle AdaLayerNorm imports for different diffusers versions
try:
    from diffusers.models.attention import AdaLayerNorm, AdaLayerNormZero
except ImportError:
    from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero


class SnakeBeta(nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
                 dropout: float = 0.0, activation_fn: str = "geglu", final_dropout: bool = False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        elif activation_fn == "snakebeta":
            act_fn = SnakeBeta(dim, inner_dim)
        else:
            act_fn = GEGLU(dim, inner_dim)

        self.net = nn.ModuleList([act_fn, nn.Dropout(dropout), LoRACompatibleLinear(inner_dim, dim_out)])
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 dropout=0.0, cross_attention_dim: Optional[int] = None,
                 activation_fn: str = "geglu", num_embeds_ada_norm: Optional[int] = None,
                 attention_bias: bool = False, only_cross_attention: bool = False,
                 double_self_attention: bool = False, upcast_attention: bool = False,
                 norm_elementwise_affine: bool = True, norm_type: str = "layer_norm",
                 final_dropout: bool = False):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim,
                               dropout=dropout, bias=attention_bias,
                               cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                               upcast_attention=upcast_attention)

        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = (AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm
                          else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine))
            self.attn2 = Attention(query_dim=dim, cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                                   heads=num_attention_heads, dim_head=attention_head_dim,
                                   dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention)
        else:
            self.norm2 = None
            self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None, class_labels=None):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs or {}
        attn_output = self.attn1(norm_hidden_states,
                                 encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                                 attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
                                 **cross_attention_kwargs)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                                     attention_mask=encoder_attention_mask, **cross_attention_kwargs)
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        return ff_output + hidden_states
