"""
Chatterbox TTS - vLLM-based implementation for Triton.
Based on https://github.com/randombk/chatterbox-vllm
"""
from .models.s3gen import S3Gen, S3GEN_SR
from .models.s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.t3_config import T3Config
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm

__all__ = [
    'S3Gen', 'S3GEN_SR', 'S3_SR', 'SPEECH_VOCAB_SIZE', 'drop_invalid_tokens',
    'VoiceEncoder', 'SPEECH_TOKEN_OFFSET', 'T3Config', 'T3Cond', 'T3CondEnc',
    'LearnedPositionEmbeddings', 'punc_norm',
]
