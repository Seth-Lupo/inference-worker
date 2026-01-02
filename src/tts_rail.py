"""
TTS Rail - Text-to-Speech synthesis with mock and real implementations.

Provides a unified interface for TTS with dependency injection support.
"""
import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Chunk of synthesized audio."""
    data: bytes
    is_final: bool


class TTSRail(Protocol):
    """Protocol defining the TTS interface."""

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Synthesize text to streaming audio chunks."""
        ...

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis."""
        ...

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently outputting audio."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...


class MockTTSRail:
    """
    Mock TTS Rail that generates a 1-second beep for any input.

    Used for testing when Triton/CosyVoice is not available.
    """

    SAMPLE_RATE = 16000
    BEEP_FREQUENCY = 440  # A4 note
    BEEP_DURATION = 1.0   # seconds

    def __init__(self):
        self._interrupted = False
        self._is_speaking = False
        logger.debug("MockTTSRail initialized")

    def _generate_beep(self) -> bytes:
        """Generate a 1-second 440Hz beep as PCM16 audio."""
        start_time = time.perf_counter()

        num_samples = int(self.SAMPLE_RATE * self.BEEP_DURATION)
        t = np.linspace(0, self.BEEP_DURATION, num_samples, dtype=np.float32)

        # Generate sine wave with fade in/out to avoid clicks
        fade_samples = int(self.SAMPLE_RATE * 0.01)  # 10ms fade
        envelope = np.ones(num_samples, dtype=np.float32)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        # Generate sine wave
        audio = np.sin(2 * np.pi * self.BEEP_FREQUENCY * t) * envelope * 0.5

        # Convert to PCM16
        audio_int16 = (audio * 32767).astype(np.int16)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Beep generated: {len(audio_int16)} samples in {elapsed_ms:.2f}ms")

        return audio_int16.tobytes()

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize text to audio (generates beep regardless of input).

        Args:
            text: Text to synthesize (ignored in mock)

        Yields:
            AudioChunk with PCM16 audio data
        """
        start_time = time.perf_counter()
        self._interrupted = False
        self._is_speaking = True

        logger.debug(f"MockTTS synthesize called with: '{text}'")

        try:
            # Generate full beep
            beep_audio = self._generate_beep()

            if not self._interrupted:
                logger.debug(f"MockTTS emitting {len(beep_audio)} bytes of audio")
                yield AudioChunk(data=beep_audio, is_final=True)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"MockTTS synthesis complete in {elapsed_ms:.2f}ms")

        finally:
            self._is_speaking = False
            logger.debug("MockTTS finished speaking")

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis (for barge-in)."""
        logger.debug("MockTTS interrupted")
        self._interrupted = True

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently outputting audio."""
        return self._is_speaking

    def reset(self) -> None:
        """Reset internal state."""
        logger.debug("MockTTS reset")
        self._interrupted = False
        self._is_speaking = False


class CosyVoiceTTSRail:
    """
    CosyVoice 2 TTS Rail using Triton Inference Server.

    Wrapper around CosyVoiceTTS that conforms to the TTSRail interface.
    """

    def __init__(
        self,
        triton_url: str = "localhost:8001",
        model_name: str = "cosyvoice2",
    ):
        """
        Initialize CosyVoice TTS.

        Args:
            triton_url: Triton gRPC endpoint
            model_name: CosyVoice model name in Triton
        """
        from .triton_client import TritonClient, TritonConfig
        from .tts.cosyvoice_tts import CosyVoiceTTS, CosyVoiceConfig

        # Parse URL
        if ":" in triton_url:
            host, port = triton_url.rsplit(":", 1)
            port = int(port)
        else:
            host = triton_url
            port = 8001

        # Create client and TTS
        triton_config = TritonConfig(host=host, port=port)
        self._client = TritonClient(triton_config)

        tts_config = CosyVoiceConfig(model_name=model_name)
        self._tts = CosyVoiceTTS(
            triton_client=self._client,
            config=tts_config,
        )

        logger.debug(f"CosyVoiceTTSRail initialized: {triton_url}/{model_name}")

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize text to streaming audio.

        Args:
            text: Text to synthesize

        Yields:
            AudioChunk with PCM16 audio data (16kHz mono)
        """
        async for chunk in self._tts.synthesize(text):
            yield chunk

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis."""
        self._tts.interrupt()

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently outputting audio."""
        return self._tts.is_speaking

    def reset(self) -> None:
        """Reset internal state."""
        self._tts.reset()


class TTSBackend(Enum):
    """Available TTS backends."""
    MOCK = "mock"
    COSYVOICE = "cosyvoice"


def create_tts_rail(
    backend: TTSBackend = TTSBackend.MOCK,
    triton_url: str = "localhost:8001",
    model_name: str = "cosyvoice2",
) -> TTSRail:
    """
    Factory function to create TTS rail.

    Args:
        backend: Which TTS backend to use
        triton_url: Triton server URL (for CosyVoice)
        model_name: Model name (for CosyVoice)

    Returns:
        TTSRail implementation
    """
    if backend == TTSBackend.MOCK:
        logger.info("Creating MockTTSRail")
        return MockTTSRail()

    elif backend == TTSBackend.COSYVOICE:
        logger.info(f"Creating CosyVoiceTTSRail: {triton_url}/{model_name}")
        return CosyVoiceTTSRail(
            triton_url=triton_url,
            model_name=model_name,
        )

    else:
        raise ValueError(f"Unknown TTS backend: {backend}")


# For backwards compatibility, expose MockTTSRail as TTSRail
# (existing code uses TTSRail() directly)
class TTSRailCompat(MockTTSRail):
    """Backwards-compatible TTSRail (defaults to mock)."""
    pass


# Alias for backwards compatibility
TTSRail = TTSRailCompat
