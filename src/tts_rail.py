"""
TTS Rail - Mock implementation that generates a beep sound.
"""
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class AudioChunk:
    data: bytes
    is_final: bool


class TTSRail:
    """
    Mock TTS Rail that generates a 1-second beep for any input.

    In production, this would use CosyVoice 2 via Triton TensorRT-LLM.
    """

    SAMPLE_RATE = 16000
    BEEP_FREQUENCY = 440  # A4 note
    BEEP_DURATION = 1.0   # seconds
    CHUNK_DURATION = 0.02  # 20ms chunks

    def __init__(self):
        self._interrupted = False
        self._is_speaking = False

    def _generate_beep(self) -> bytes:
        """Generate a 1-second 440Hz beep as PCM16 audio."""
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
        return audio_int16.tobytes()

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize (ignored in mock)

        Yields:
            AudioChunk with PCM16 audio data
        """
        self._interrupted = False
        self._is_speaking = True

        try:
            # Generate full beep
            beep_audio = self._generate_beep()

            # Stream in chunks
            chunk_size = int(self.SAMPLE_RATE * self.CHUNK_DURATION * 2)  # *2 for PCM16
            num_chunks = len(beep_audio) // chunk_size

            for i in range(num_chunks + 1):
                if self._interrupted:
                    return

                start = i * chunk_size
                end = min(start + chunk_size, len(beep_audio))

                if start >= len(beep_audio):
                    break

                chunk_data = beep_audio[start:end]
                is_final = end >= len(beep_audio)

                yield AudioChunk(data=chunk_data, is_final=is_final)

                # Simulate real-time streaming
                await asyncio.sleep(self.CHUNK_DURATION)

        finally:
            self._is_speaking = False

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis (for barge-in)."""
        self._interrupted = True

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently outputting audio."""
        return self._is_speaking

    def reset(self) -> None:
        """Reset internal state."""
        self._interrupted = False
        self._is_speaking = False
