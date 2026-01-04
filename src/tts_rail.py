"""
TTS Rail - Text-to-Speech abstraction with mock and CosyVoice backends.

Supports:
- MockTTSRail: Generates a beep for testing
- CosyVoiceTTSRail: CosyVoice 2 via Triton Inference Server
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator
import numpy as np

logger = logging.getLogger(__name__)

# Sampling rate constants
SAMPLE_RATE = 16000  # Target output sample rate (PCM16, mono)


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
    CosyVoice 2 TTS via Triton Inference Server.

    Streams audio chunks as they're generated for low latency.
    Supports interrupt for barge-in.
    Supports voice cloning via reference audio.
    """

    def __init__(
        self,
        triton_url: str = "localhost:8001",
        model_name: str = "cosyvoice2",
        reference_audio_path: str = None,
        reference_text: str = None,
    ):
        from .triton_client import TritonClient, TritonConfig
        from .tts.cosyvoice_tts import CosyVoiceTTS, TTSConfig

        # Parse URL
        host, port = triton_url.rsplit(":", 1) if ":" in triton_url else (triton_url, "8001")

        logger.info(f"CosyVoiceTTSRail: connecting to {host}:{port}, model={model_name}")
        if reference_audio_path:
            logger.info(f"CosyVoiceTTSRail: using voice cloning from {reference_audio_path}")

        # Create client
        self._client = TritonClient(TritonConfig(host=host, port=int(port)))
        self._tts = CosyVoiceTTS(
            self._client,
            TTSConfig(
                model_name=model_name,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
            )
        )
        self._connected = False
        self._triton_url = triton_url
        self._model_name = model_name

    async def _ensure_connected(self) -> None:
        """Lazy connection to Triton."""
        if not self._connected:
            logger.info(f"TTS: Connecting to Triton at {self._triton_url}...")
            connect_start = time.perf_counter()
            await self._client.connect()
            self._connected = True
            elapsed = (time.perf_counter() - connect_start) * 1000
            logger.info(f"TTS: Connected to Triton in {elapsed:.0f}ms")

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize text to streaming audio.

        Args:
            text: Text to synthesize

        Yields:
            AudioChunk with PCM16 audio data
        """
        if not text.strip():
            logger.warning("TTS: Empty text, skipping synthesis")
            return

        logger.info(f"TTS: Synthesizing {len(text)} chars: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        synth_start = time.perf_counter()

        await self._ensure_connected()

        chunk_count = 0
        total_bytes = 0
        first_chunk_time = None

        try:
            async for chunk in self._tts.synthesize(text):
                chunk_count += 1
                chunk_bytes = len(chunk.data)
                total_bytes += chunk_bytes

                if chunk_count == 1:
                    first_chunk_time = (time.perf_counter() - synth_start) * 1000
                    logger.info(f"TTS: First chunk in {first_chunk_time:.0f}ms ({chunk_bytes} bytes)")

                logger.debug(f"TTS: Chunk {chunk_count}: {chunk_bytes} bytes, final={chunk.is_final}")

                yield AudioChunk(data=chunk.data, is_final=chunk.is_final)

            # Summary
            total_time = (time.perf_counter() - synth_start) * 1000
            duration_ms = (total_bytes / 2) / SAMPLE_RATE * 1000  # PCM16 = 2 bytes/sample
            rtf = total_time / duration_ms if duration_ms > 0 else 0

            logger.info(
                f"TTS: Complete - {chunk_count} chunks, {total_bytes} bytes "
                f"({duration_ms:.0f}ms audio) in {total_time:.0f}ms (RTF={rtf:.2f})"
            )

        except Exception as e:
            logger.error(f"TTS: Synthesis error: {e}", exc_info=True)
            raise

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis (for barge-in)."""
        logger.info("TTS: Interrupted")
        self._tts.interrupt()

    @property
    def is_speaking(self) -> bool:
        return self._tts.is_speaking

    def reset(self) -> None:
        """Reset TTS state."""
        logger.debug("TTS: Reset")
        self._tts.reset()


class TTSBackend(Enum):
    """Available TTS backends."""
    MOCK = "mock"
    COSYVOICE = "cosyvoice"


def create_tts_rail(
    backend: TTSBackend = TTSBackend.MOCK,
    triton_url: str = "localhost:8001",
    model_name: str = "cosyvoice2",
    reference_audio_path: str = None,
    reference_text: str = None,
) -> BaseTTSRail:
    """
    Factory to create TTS rail.

    Args:
        backend: Which backend to use
        triton_url: Triton URL (for CosyVoice)
        model_name: Model name (for CosyVoice)
        reference_audio_path: Path to reference audio for voice cloning
        reference_text: Text spoken in reference audio
    """
    if backend == TTSBackend.MOCK:
        logger.info("Using MockTTSRail")
        return MockTTSRail()

    elif backend == TTSBackend.COSYVOICE:
        logger.info(f"Using CosyVoiceTTSRail: {triton_url}/{model_name}")
        return CosyVoiceTTSRail(
            triton_url=triton_url,
            model_name=model_name,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )

    raise ValueError(f"Unknown backend: {backend}")


# Backwards compatibility alias
TTSRail = MockTTSRail
