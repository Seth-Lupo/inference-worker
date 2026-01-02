"""
ASR Rail - Mock implementation that returns fixed text.
"""
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


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

    async def transcribe(self, audio_data: bytes) -> AsyncIterator[TranscriptEvent]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: PCM16 audio bytes

        Yields:
            TranscriptEvent with transcription
        """
        # Simulate some processing delay
        await asyncio.sleep(0.05)

        # Return mock transcription
        yield TranscriptEvent(
            text="This is ASR Test",
            is_final=True
        )

    def reset(self) -> None:
        """Reset internal state."""
        self._buffer = b""
