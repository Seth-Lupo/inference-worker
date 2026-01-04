"""T3 Model for vLLM - Chatterbox Speech Token Generator.

This model generates discrete speech tokens from text input.
Conditioning (speaker embedding) is passed via the multimodal "image" input
as base64-encoded tensor data for compatibility with Triton's vLLM backend.
"""
import base64
import io
import os
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PreTrainedModel

from vllm.attention import Attention, AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalDataDict,
    MultiModalFieldConfig,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .configuration_t3 import T3Config

# Speech token offset for distinguishing prefill vs decode tokens
SPEECH_TOKEN_OFFSET = 2500
CONDITIONING_SIZE = 34


class T3ProcessingInfo(BaseProcessingInfo):
    """Processing info for T3 multimodal inputs."""

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Use "image" key for compatibility with Triton vLLM backend
        return {"image": 1}


class T3DummyInputsBuilder(BaseDummyInputsBuilder):
    """Dummy inputs for profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "[START]Hello world[STOP]"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        # Dummy conditioning tensor
        dummy_cond = torch.zeros(CONDITIONING_SIZE, 1024)
        return {"image": [dummy_cond]}


class T3MultiModalProcessor(BaseMultiModalProcessor):
    """Process multimodal inputs for T3.

    Conditioning is passed via "image" input for Triton vLLM backend compatibility.
    The data can be:
    - A torch.Tensor (direct conditioning embeddings)
    - A base64-encoded string (serialized conditioning from Triton)
    """

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {}

    def _get_prompt_updates(
        self,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        mm_processor_kwargs: Mapping[str, object],
    ) -> Mapping[str, PromptUpdate]:
        return {}

    def apply(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Tuple:
        # Process conditioning from "image" input
        images = mm_data.get("image", [])
        conditioning = None

        if images:
            img_data = images[0]
            if isinstance(img_data, torch.Tensor):
                conditioning = img_data
            elif isinstance(img_data, str):
                # Base64-encoded tensor from Triton
                try:
                    decoded = base64.b64decode(img_data)
                    buffer = io.BytesIO(decoded)
                    conditioning = torch.load(buffer, weights_only=True)
                except Exception:
                    conditioning = None
            elif isinstance(img_data, bytes):
                try:
                    buffer = io.BytesIO(img_data)
                    conditioning = torch.load(buffer, weights_only=True)
                except Exception:
                    conditioning = None

        return prompt, {"conditioning": conditioning}


class T3MLP(nn.Module):
    """T3 MLP layer (SwiGLU)."""

    def __init__(self, config: T3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=config.mlp_bias,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class T3Attention(nn.Module):
    """T3 Attention layer."""

    def __init__(self, config: T3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.num_heads // self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class T3DecoderLayer(nn.Module):
    """T3 Transformer Decoder Layer."""

    def __init__(self, config: T3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = T3Attention(config, layer_idx)
        self.mlp = T3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class T3Model(nn.Module):
    """T3 Transformer Model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            T3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Load pre-computed conditioning if available
        self.conditioning = None
        if hasattr(config, 'conditioning_path') and config.conditioning_path:
            try:
                if os.path.exists(config.conditioning_path):
                    self.conditioning = torch.load(
                        config.conditioning_path, weights_only=True
                    )
                    print(f"T3: Loaded pre-computed conditioning from {config.conditioning_path}")
            except Exception as e:
                print(f"T3: Failed to load conditioning: {e}")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    T3MultiModalProcessor,
    info=T3ProcessingInfo,
    dummy_inputs=T3DummyInputsBuilder,
)
class T3ForCausalLM(nn.Module):
    """T3 for Causal Language Modeling - Speech Token Generation.

    This model generates speech tokens from text, with optional speaker conditioning.
    For use with Triton's vLLM backend, conditioning is passed via "image" input.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = T3Model(vllm_config=vllm_config, prefix=prefix)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        """Get conditioning embeddings from multimodal input."""
        conditioning = kwargs.get("conditioning", None)

        if conditioning is None:
            # Fall back to pre-computed conditioning
            conditioning = self.model.conditioning

        if conditioning is not None:
            if not isinstance(conditioning, torch.Tensor):
                conditioning = torch.tensor(conditioning)
            return [conditioning.unsqueeze(0)]

        return None

    def get_input_processor(self):
        return T3MultiModalProcessor(self.config, self.vllm_config.model_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Get conditioning embeddings if available
        mm_embeddings = self.get_multimodal_embeddings(**kwargs)

        if mm_embeddings is not None and inputs_embeds is None:
            # Prepend conditioning to input embeddings
            text_embeds = self.get_input_embeddings(input_ids)
            cond_embeds = mm_embeddings[0].to(text_embeds.device, text_embeds.dtype)
            inputs_embeds = torch.cat([cond_embeds, text_embeds], dim=0)
            # Adjust positions
            cond_len = cond_embeds.shape[0]
            positions = positions + cond_len

        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        """Load model weights from safetensors."""
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded = set()

        for name, loaded_weight in weights:
            # Handle name mapping from T3 weights to vLLM format
            if name.startswith("lm."):
                name = name[3:]  # Remove "lm." prefix

            # Map layer names
            if "layers." in name:
                name = name.replace("layers.", "model.layers.")

            if "embed_tokens" in name:
                name = name.replace("embed_tokens", "model.embed_tokens")

            if "norm." in name and "layernorm" not in name:
                name = name.replace("norm.", "model.norm.")

            if "lm_head" in name:
                name = "lm_head.weight"

            # Handle stacked params
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    name = name.replace(weight_name, param_name)
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                        loaded.add(name)
                    break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded.add(name)

        return loaded


# Register T3 with vLLM's model registry for custom model support
try:
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model("T3ForCausalLM", T3ForCausalLM)
except (ImportError, AttributeError):
    # Fallback for different vLLM versions
    try:
        from vllm.model_executor.models import _MODELS
        _MODELS["T3ForCausalLM"] = T3ForCausalLM
    except ImportError:
        pass
