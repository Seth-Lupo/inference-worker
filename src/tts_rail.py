"""
TTS Rail - Mock implementation that generates a beep sound.
"""
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import AsyncIterator

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self._interrupted = False
        self._is_speaking = False
        logger.debug("TTSRail initialized")

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
        Synthesize text to audio.

        Args:
            text: Text to synthesize (ignored in mock)

        Yields:
            AudioChunk with PCM16 audio data
        """
        start_time = time.perf_counter()
        self._interrupted = False
        self._is_speaking = True

        logger.debug(f"TTS synthesize called with: '{text}'")

        try:
            # Generate full beep
            beep_audio = self._generate_beep()

            # Send all audio at once for minimal latency
            # (In production, you'd stream chunks as they're generated)
            logger.debug(f"TTS emitting {len(beep_audio)} bytes of audio")

            yield AudioChunk(data=beep_audio, is_final=True)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"TTS synthesis complete in {elapsed_ms:.2f}ms")

        finally:
            self._is_speaking = False
            logger.debug("TTS finished speaking")

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis (for barge-in)."""
        logger.debug("TTS interrupted")
        self._interrupted = True

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently outputting audio."""
        return self._is_speaking

    def reset(self) -> None:
        """Reset internal state."""
        logger.debug("TTS reset")
        self._interrupted = False
        self._is_speaking = False
