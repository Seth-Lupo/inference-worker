"""
TTS (Text-to-Speech) module.

Provides TTSRail implementations:
- MockTTS: Generates beep sounds for testing
- CosyVoiceTTS: Real TTS via Triton/CosyVoice2
"""
from .cosyvoice_tts import CosyVoiceTTS, CosyVoiceConfig, AudioChunk

__all__ = [
    "CosyVoiceTTS",
    "CosyVoiceConfig",
    "AudioChunk",
]
