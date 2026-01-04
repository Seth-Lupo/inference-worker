"""T3 vLLM Model - Chatterbox Speech Token Generator.
Based on https://github.com/randombk/chatterbox-vllm
"""
from typing import Iterable, Mapping, Optional, Sequence, Union
import os
import random

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, MultiModalKwargsItem, MultiModalBatchedField
from vllm.multimodal.parse import MultiModalDataParser, ModalityDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalDataDict,
    MultiModalDataItems,
    MultiModalFieldConfig,
    PromptUpdate,
    MultiModalInputs,
    PlaceholderRange,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .modules.learned_pos_emb import LearnedPositionEmbeddings
from .modules.t3_config import T3Config
from .modules.cond_enc import T3Cond, T3CondEnc


PREFILL_COND_START_TOKEN = 695
PREFILL_COND_END_TOKEN = 696
PREFILL_END_TOKEN = 697
CONDITIONING_SIZE = 34
SPEECH_TOKEN_OFFSET = 2500


class T3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"conditionals": 1}


class T3MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "[START]Hello, world![STOP]"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return {"conditionals": [torch.zeros(CONDITIONING_SIZE, 2048)] * mm_counts["conditionals"]}


class T3MultiModalDataParser(MultiModalDataParser):
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        conditionals = mm_data.get("conditionals", None)
        if conditionals is None:
            return MultiModalDataItems({})
        return MultiModalDataItems({"conditionals": ConditionalsEmbeddingItems(conditionals)})


class ConditionalsEmbeddingItems(ModalityDataItems[torch.Tensor, torch.Tensor]):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(data, "conditionals")

    def get_count(self) -> int:
        return 1

    def get(self, index: int) -> torch.Tensor:
        assert index == 0
        return self.data

    def get_processor_data(self) -> Mapping[str, torch.Tensor]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, torch.Tensor]:
        return {"conditionals": self.data}


def create_triangular_matrix(m, n):
    row_indices = torch.arange(m).unsqueeze(1)
    col_indices = torch.arange(n).unsqueeze(0)
    matrix = (col_indices <= row_indices).float()
    return matrix


class T3MultiModalProcessor(BaseMultiModalProcessor[T3ProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return T3MultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(conditionals=MultiModalFieldConfig.batched("conditionals"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        return []

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        processed_outputs = tokenizer(prompt, return_tensors="pt")
        processed_outputs['conditionals'] = mm_data.get('conditionals', None)
        return processed_outputs

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        mm_items = self._to_mm_items(mm_data)
        (prompt_ids, mm_kwargs, mm_hashes, is_update_applied) = self._apply_hf_processor(
            prompt, mm_items, hf_processor_mm_kwargs, tokenization_kwargs, return_mm_hashes=False,
        )

        final_prompt_ids = [
            PREFILL_COND_START_TOKEN,
            *([prompt_ids[0]] * (CONDITIONING_SIZE-2)),
            PREFILL_COND_END_TOKEN,
            *prompt_ids,
            PREFILL_END_TOKEN,
        ]

        conditionals = mm_data.get("conditionals", None)
        assert conditionals is not None and len(conditionals) > 0
        assert len(conditionals) == 1
        assert conditionals[0].shape[0] == CONDITIONING_SIZE

        new_conditionals = torch.cat([
            conditionals[0],
            create_triangular_matrix(len(prompt_ids), conditionals[0].shape[1]).to(conditionals[0].device),
            torch.zeros(1, conditionals[0].shape[1]).to(conditionals[0].device),
        ], dim=0)

        new_mm_kwargs = MultiModalKwargs.from_items([
            MultiModalKwargsItem.from_elems(
                MultiModalBatchedField().build_elems(
                    modality="conditionals",
                    key="conditionals",
                    data=[new_conditionals],
                )
            )
        ])

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=final_prompt_ids,
            mm_kwargs=new_mm_kwargs,
            mm_hashes={"conditionals": [str(random.random())]},
            mm_placeholders={"conditionals": [PlaceholderRange(offset=0, length=len(final_prompt_ids), is_embed=None)]},
        )


@MULTIMODAL_REGISTRY.register_processor(T3MultiModalProcessor,
                                        info=T3ProcessingInfo,
                                        dummy_inputs=T3MultiModalDummyInputsBuilder)
class T3ForCausalLM(nn.Module, VllmModelForTextGeneration, SupportsMultiModal):
    """Native vLLM implementation of Chatterbox T3."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        vllm_config.model_config.hf_config.hidden_size = 1024
        self.vllm_config = vllm_config
        self.cfg: ModelConfig = vllm_config.model_config

        self.tfmr = LlamaModel(vllm_config=vllm_config, prefix=prefix + ".tfmr")

        text_tokens_dict_size = 704

        self.t3conf = T3Config()
        self.dim = self.t3conf.n_channels
        self.cond_enc = T3CondEnc(self.t3conf)
        self.text_emb = nn.Embedding(text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.t3conf.speech_tokens_dict_size, self.dim)

        max_text_seq_len = self.t3conf.max_text_tokens + 2
        self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

        max_mel_seq_len = self.t3conf.max_speech_tokens + 2 + 2
        self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        self.speech_head = ParallelLMHead(
            num_embeddings=self.t3conf.speech_tokens_dict_size,
            embedding_dim=self.dim,
            padding_size=1,
            prefix=prefix + ".speech_head",
        )
        self.logits_processor = LogitsProcessor(self.t3conf.speech_tokens_dict_size)

        self.cfg_scale = float(os.environ.get("CHATTERBOX_CFG_SCALE", "0.5"))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        state_dicts = {}
        hf_llama_weights = {}

        for name, weight in weights:
            if name.startswith("tfmr."):
                subname = name[5:]
                hf_llama_weights[subname] = weight
                continue
            loaded_params.add(name)
            attr, subname = name.split('.', 1)
            state_dict = state_dicts.get(attr, {})
            state_dict[subname] = weight
            state_dicts[attr] = state_dict

        for attr, state_dict in state_dicts.items():
            if hasattr(self, attr):
                getattr(self, attr).load_state_dict(state_dict)

        llama_loaded_params = self.tfmr.load_weights(hf_llama_weights.items())
        loaded_params.update('tfmr.' + i for i in llama_loaded_params)

        text_position_ids = torch.arange(self.t3conf.max_text_tokens + 2, device=self.text_pos_emb.emb.weight.device)
        self.precomputed_text_pos_emb = self.text_pos_emb.get_fixed_embedding(text_position_ids)[0]

        speech_position_ids = torch.arange(self.t3conf.max_speech_tokens + 2 + 2, device=self.speech_pos_emb.emb.weight.device)
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(speech_position_ids)[0]

        return loaded_params

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        conditionals = kwargs.get("conditionals", [])
        return [batch[0] for batch in conditionals]

    def split_prefill_decode(self, input_ids: torch.Tensor, multimodal_embeddings: list) -> list:
        if len(input_ids) == 0:
            return []

        remaining_multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)
        in_prefill_block = input_ids[0] < SPEECH_TOKEN_OFFSET
        output = []
        buffer = []

        for input_id in input_ids:
            if (in_prefill_block != (input_id < SPEECH_TOKEN_OFFSET)) or (input_id == PREFILL_COND_START_TOKEN):
                if buffer:
                    if in_prefill_block:
                        mme, remaining_multimodal_embeddings = remaining_multimodal_embeddings.split(
                            [len(buffer), len(remaining_multimodal_embeddings) - len(buffer)], dim=0)
                        output.append((torch.tensor(buffer).to(input_ids.device), mme))
                    else:
                        output.append((torch.tensor(buffer).to(input_ids.device), None))
                buffer = []
                in_prefill_block = (input_id < SPEECH_TOKEN_OFFSET)
            buffer.append(input_id)

        if buffer:
            if in_prefill_block:
                mme, remaining_multimodal_embeddings = remaining_multimodal_embeddings.split(
                    [len(buffer), len(remaining_multimodal_embeddings) - len(buffer)], dim=0)
                output.append((torch.tensor(buffer).to(input_ids.device), mme))
            else:
                output.append((torch.tensor(buffer).to(input_ids.device), None))

        return output

    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings=None) -> torch.Tensor:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            embeds = self.speech_emb(input_ids - SPEECH_TOKEN_OFFSET)
            return torch.cat([embeds, embeds], dim=1)

        out = []
        for ids, multimodal_embedding in self.split_prefill_decode(input_ids, multimodal_embeddings):
            if multimodal_embedding is None:
                embeds = self.speech_emb(ids - SPEECH_TOKEN_OFFSET)
                out.append(torch.cat([embeds, embeds], dim=1))
                continue

            if ids[0] == PREFILL_COND_START_TOKEN and ids[-1] == PREFILL_END_TOKEN:
                text_ids = ids[CONDITIONING_SIZE:-1]
                text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]
                start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1]
                conditioning_emb = multimodal_embedding[0:CONDITIONING_SIZE]
                cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))

            elif ids[0] == PREFILL_COND_START_TOKEN:
                text_ids = ids[CONDITIONING_SIZE:]
                text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]
                conditioning_emb = multimodal_embedding[0:min(CONDITIONING_SIZE, len(multimodal_embedding))]
                cond_embeds = torch.cat([conditioning_emb, text_emb], dim=0)
                uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb)], dim=0)
                out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))

            elif ids[-1] == PREFILL_END_TOKEN:
                indices = torch.where(ids == PREFILL_COND_END_TOKEN)[0]
                if len(indices) > 0:
                    text_ids = ids[indices[0]+1:-1]
                    text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]
                    start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                    start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1]
                    conditioning_emb = multimodal_embedding[:indices[0]+1]
                    cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                    uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                    out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))
                else:
                    text_ids = ids[:-1]
                    text_pos = torch.sum(multimodal_embedding[0:len(text_ids)], dim=1) - 1
                    text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[text_pos.tolist()]
                    start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                    start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1]
                    cond_embeds = torch.cat([text_emb, start_of_speech_emb], dim=0)
                    uncond_embeds = torch.cat([torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                    out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))
            else:
                raise ValueError(f"Unknown prefill block: {ids}")

        return torch.cat(out, dim=0)

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        cond_hidden_states, uncond_hidden_states = hidden_states.split([self.dim, self.dim], dim=1)
        cond_logits = self.logits_processor(self.speech_head, cond_hidden_states, sampling_metadata)
        uncond_logits = self.logits_processor(self.speech_head, uncond_hidden_states, sampling_metadata)
        logits = cond_logits + self.cfg_scale * (cond_logits - uncond_logits)
        logits = torch.cat([
            torch.zeros(logits.shape[0], SPEECH_TOKEN_OFFSET).to(logits.device).fill_(float('-inf')),
            logits,
        ], dim=1)
        return logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, [])

        cond_embeds, uncond_embeds = inputs_embeds.split([self.dim, self.dim], dim=1)

        hidden_states = self.tfmr(
            input_ids=None,
            positions=torch.cat([positions, positions], dim=0),
            intermediate_tensors=None,
            inputs_embeds=torch.cat([cond_embeds, uncond_embeds], dim=0)
        )

        hidden_state_1, hidden_state_2 = hidden_states.split([len(cond_embeds), len(uncond_embeds)], dim=0)
        return torch.cat([hidden_state_1, hidden_state_2], dim=1)

    def get_language_model(self) -> torch.nn.Module:
        return self.tfmr
