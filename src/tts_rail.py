"""
TTS Rail - Text-to-Speech abstraction with mock and CosyVoice backends.
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Chunk of synthesized audio."""
    data: bytes      # PCM16 audio bytes
    is_final: bool   # True if last chunk


class BaseTTSRail(ABC):
    """Base class for TTS implementations."""

    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Synthesize text to streaming audio chunks."""
        pass

    @abstractmethod
    def interrupt(self) -> None:
        """Interrupt ongoing synthesis."""
        pass

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Whether TTS is outputting audio."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        pass


class MockTTSRail(BaseTTSRail):
    """
    Mock TTS that generates a short beep for testing.
    """

    SAMPLE_RATE = 16000
    BEEP_FREQ = 440
    BEEP_DURATION = 0.5

    def __init__(self):
        self._interrupted = False
        self._is_speaking = False

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Generate a beep regardless of input text."""
        self._interrupted = False
        self._is_speaking = True

        logger.debug(f"MockTTS: '{text[:30]}...'")

        try:
            if not self._interrupted:
                audio = self._generate_beep()
                yield AudioChunk(data=audio, is_final=True)
        finally:
            self._is_speaking = False

    def _generate_beep(self) -> bytes:
        """Generate sine wave beep."""
        samples = int(self.SAMPLE_RATE * self.BEEP_DURATION)
        t = np.linspace(0, self.BEEP_DURATION, samples, dtype=np.float32)

        # Sine wave with envelope
        envelope = np.ones(samples, dtype=np.float32)
        fade = int(self.SAMPLE_RATE * 0.01)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)

        audio = np.sin(2 * np.pi * self.BEEP_FREQ * t) * envelope * 0.5
        return (audio * 32767).astype(np.int16).tobytes()

    def interrupt(self) -> None:
        self._interrupted = True

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        self._interrupted = False
        self._is_speaking = False


class CosyVoiceTTSRail(BaseTTSRail):
    """
    CosyVoice 2 TTS via Triton.
    """

    def __init__(self, triton_url: str = "localhost:8001", model_name: str = "cosyvoice2"):
        from .triton_client import TritonClient, TritonConfig
        from .tts.cosyvoice_tts import CosyVoiceTTS, TTSConfig

        # Parse URL
        host, port = triton_url.rsplit(":", 1) if ":" in triton_url else (triton_url, "8001")

        # Create client
        self._client = TritonClient(TritonConfig(host=host, port=int(port)))
        self._tts = CosyVoiceTTS(self._client, TTSConfig(model_name=model_name))
        self._connected = False

        logger.debug(f"CosyVoiceTTSRail: {triton_url}/{model_name}")

    async def _ensure_connected(self) -> None:
        """Lazy connection to Triton."""
        if not self._connected:
            await self._client.connect()
            self._connected = True

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Synthesize text to streaming audio."""
        await self._ensure_connected()

        async for chunk in self._tts.synthesize(text):
            yield AudioChunk(data=chunk.data, is_final=chunk.is_final)

    def interrupt(self) -> None:
        self._tts.interrupt()

    @property
    def is_speaking(self) -> bool:
        return self._tts.is_speaking

    def reset(self) -> None:
        self._tts.reset()


class TTSBackend(Enum):
    """Available TTS backends."""
    MOCK = "mock"
    COSYVOICE = "cosyvoice"


def create_tts_rail(
    backend: TTSBackend = TTSBackend.MOCK,
    triton_url: str = "localhost:8001",
    model_name: str = "cosyvoice2",
) -> BaseTTSRail:
    """
    Factory to create TTS rail.

    Args:
        backend: Which backend to use
        triton_url: Triton URL (for CosyVoice)
        model_name: Model name (for CosyVoice)
    """
    if backend == TTSBackend.MOCK:
        logger.info("Using MockTTSRail")
        return MockTTSRail()

    elif backend == TTSBackend.COSYVOICE:
        logger.info(f"Using CosyVoiceTTSRail: {triton_url}/{model_name}")
        return CosyVoiceTTSRail(triton_url=triton_url, model_name=model_name)

    raise ValueError(f"Unknown backend: {backend}")


# Backwards compatibility alias
TTSRail = MockTTSRail
