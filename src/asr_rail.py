"""
ASR Rail - Mock implementation that returns fixed text.
"""
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool


class ASRRail:
    """
    Mock ASR Rail that returns "This is ASR Test" for any audio input.

    In production, this would use Parakeet TDT via Triton.
    """

    def __init__(self):
        self._buffer: bytes = b""
        logger.debug("ASRRail initialized")

    async def transcribe(self, audio_data: bytes) -> AsyncIterator[TranscriptEvent]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: PCM16 audio bytes

        Yields:
            TranscriptEvent with transcription
        """
        start_time = time.perf_counter()
        audio_duration_ms = (len(audio_data) / 2) / 16  # PCM16 = 2 bytes/sample, 16kHz

        logger.debug(f"ASR transcribe called: {len(audio_data)} bytes ({audio_duration_ms:.0f}ms of audio)")

        # Mock transcription - no delay
        response = "This is ASR Test"

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"ASR emitting transcript: '{response}' ({elapsed_ms:.2f}ms)")

        yield TranscriptEvent(text=response, is_final=True)

        logger.info(f"ASR transcription complete in {elapsed_ms:.2f}ms")

    def reset(self) -> None:
        """Reset internal state."""
        logger.debug("ASR reset")
        self._buffer = b""
