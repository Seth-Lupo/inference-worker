"""
Silero VAD wrapper for voice activity detection.
"""
import logging
import time
import torch
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class VADState(Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class VADEvent:
    state: VADState
    confidence: float


class SileroVAD:
    """
    Wrapper for Silero VAD model.

    Detects voice activity and triggers events when speech starts/ends.
    """

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512  # 32ms at 16kHz

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
    ):
        """
        Initialize Silero VAD.

        Args:
            threshold: VAD probability threshold (0-1)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Silence duration to detect end of speech
        """
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_duration_ms * self.SAMPLE_RATE / 1000)
        self.min_silence_samples = int(min_silence_duration_ms * self.SAMPLE_RATE / 1000)

        # Load Silero VAD model
        logger.debug("Loading Silero VAD model...")
        load_start = time.perf_counter()
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        self.model.eval()
        logger.info(f"Silero VAD loaded in {(time.perf_counter() - load_start)*1000:.1f}ms")

        # State tracking
        self._state = VADState.IDLE
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_probability = 0.0

    def reset(self) -> None:
        """Reset VAD state for new session."""
        logger.debug("VAD reset")
        self.model.reset_states()
        self._state = VADState.IDLE
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_probability = 0.0

    def process_chunk(self, audio_chunk: bytes) -> Optional[VADEvent]:
        """
        Process an audio chunk and return VAD event if state changed.

        Args:
            audio_chunk: PCM16 audio bytes (16kHz, mono)

        Returns:
            VADEvent if state changed, None otherwise
        """
        # Convert bytes to float tensor
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float)

        # Run VAD
        start_time = time.perf_counter()
        with torch.no_grad():
            probability = self.model(audio_tensor, self.SAMPLE_RATE).item()
        inference_ms = (time.perf_counter() - start_time) * 1000

        self._last_probability = probability
        is_speech = probability >= self.threshold
        chunk_samples = len(audio_int16)

        logger.debug(f"VAD: prob={probability:.3f} speech={is_speech} state={self._state.value} ({inference_ms:.2f}ms)")

        # State machine
        previous_state = self._state

        if self._state == VADState.IDLE:
            if is_speech:
                self._speech_samples += chunk_samples
                if self._speech_samples >= self.min_speech_samples:
                    self._state = VADState.SPEAKING
                    self._silence_samples = 0
            else:
                self._speech_samples = 0

        elif self._state == VADState.SPEAKING:
            if is_speech:
                self._silence_samples = 0
            else:
                self._silence_samples += chunk_samples
                if self._silence_samples >= self.min_silence_samples:
                    self._state = VADState.END_OF_SPEECH

        elif self._state == VADState.END_OF_SPEECH:
            # Reset for next utterance
            self._state = VADState.IDLE
            self._speech_samples = 0
            self._silence_samples = 0

        # Return event if state changed
        if self._state != previous_state:
            logger.info(f"VAD state change: {previous_state.value} -> {self._state.value} (prob={probability:.3f})")
            return VADEvent(state=self._state, confidence=probability)

        return None

    @property
    def state(self) -> VADState:
        """Current VAD state."""
        return self._state

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently detected."""
        return self._state == VADState.SPEAKING

    @property
    def last_probability(self) -> float:
        """Last VAD probability."""
        return self._last_probability
