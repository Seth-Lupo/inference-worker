"""TTS module - CosyVoice 2 via Triton."""
from .cosyvoice_tts import CosyVoiceTTS, TTSConfig, AudioChunk, create_cosyvoice_tts

__all__ = ["CosyVoiceTTS", "TTSConfig", "AudioChunk", "create_cosyvoice_tts"]
