"""
CosyVoice 2 TTS implementation using Triton Inference Server.

Provides streaming text-to-speech synthesis with voice cloning support.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import numpy as np

from ..triton_client import TritonClient, TritonConfig, get_triton_client

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Chunk of synthesized audio."""
    data: bytes
    is_final: bool


@dataclass
class CosyVoiceConfig:
    """Configuration for CosyVoice TTS."""
    model_name: str = "cosyvoice2"
    sample_rate: int = 22050  # CosyVoice native sample rate
    output_sample_rate: int = 16000  # Resample to match pipeline
    # Chunk timing (in tokens, affects latency)
    initial_chunk_tokens: int = 4
    max_chunk_tokens: int = 64
    # Voice cloning (optional)
    default_voice_id: Optional[str] = None


class CosyVoiceTTS:
    """
    CosyVoice 2 TTS via Triton Inference Server.

    Uses the cosyvoice2 BLS model which orchestrates:
    - speaker_embedding: Extract voice characteristics
    - tensorrt_llm: Generate audio tokens
    - token2wav: Convert tokens to waveform

    Supports:
    - Streaming synthesis (decoupled mode)
    - Voice cloning with reference audio
    - Barge-in interruption
    """

    def __init__(
        self,
        triton_client: Optional[TritonClient] = None,
        config: Optional[CosyVoiceConfig] = None,
    ):
        """
        Initialize CosyVoice TTS.

        Args:
            triton_client: Triton client (uses global if None)
            config: TTS configuration
        """
        self._client = triton_client
        self._use_global_client = triton_client is None
        self.config = config or CosyVoiceConfig()

        self._interrupted = False
        self._is_speaking = False
        self._current_task: Optional[asyncio.Task] = None

        # Resampling state (if output_sample_rate != sample_rate)
        self._needs_resample = (
            self.config.output_sample_rate != self.config.sample_rate
        )

        logger.debug(
            f"CosyVoiceTTS initialized: model={self.config.model_name}, "
            f"resample={self._needs_resample}"
        )

    async def _get_client(self) -> TritonClient:
        """Get the Triton client, connecting if needed."""
        if self._use_global_client:
            return await get_triton_client()
        return self._client

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses simple linear interpolation for speed.
        For better quality, use librosa or scipy.
        """
        if from_sr == to_sr:
            return audio

        # Calculate resampling ratio
        ratio = to_sr / from_sr
        new_length = int(len(audio) * ratio)

        # Linear interpolation
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled.astype(audio.dtype)

    def _float_to_pcm16(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio [-1, 1] to PCM16 bytes."""
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
    ) -> AsyncIterator[AudioChunk]:
        """
        Synthesize text to streaming audio.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier (for cached embeddings)
            reference_audio: Reference audio for voice cloning (PCM16, 16kHz)
            reference_text: Transcript of reference audio

        Yields:
            AudioChunk with PCM16 audio data (16kHz mono)
        """
        start_time = time.perf_counter()
        self._interrupted = False
        self._is_speaking = True

        logger.debug(f"TTS synthesize: '{text[:50]}...' ({len(text)} chars)")

        try:
            client = await self._get_client()

            # Check model availability
            if not await client.is_model_ready(self.config.model_name):
                raise RuntimeError(
                    f"Model {self.config.model_name} is not ready"
                )

            # Build inputs
            inputs = {
                "target_text": np.array([[text]], dtype=object),
            }

            # Add voice cloning inputs if provided
            if reference_audio is not None:
                # Convert PCM16 bytes to float32
                ref_audio = np.frombuffer(reference_audio, dtype=np.int16)
                ref_audio = ref_audio.astype(np.float32) / 32767.0
                inputs["reference_wav"] = ref_audio.reshape(1, -1)
                inputs["reference_wav_len"] = np.array(
                    [[len(ref_audio)]], dtype=np.int32
                )

            if reference_text is not None:
                inputs["reference_text"] = np.array(
                    [[reference_text]], dtype=object
                )

            # Stream inference
            total_samples = 0
            chunk_count = 0
            first_chunk_time = None

            async for result in client.infer_stream(
                model_name=self.config.model_name,
                inputs=inputs,
                output_names=["waveform"],
            ):
                # Check for interruption
                if self._interrupted:
                    logger.debug("TTS interrupted during streaming")
                    break

                # Extract waveform
                if "waveform" not in result.outputs:
                    continue

                waveform = result.outputs["waveform"]

                # Handle batch dimension
                if waveform.ndim > 1:
                    waveform = waveform.squeeze()

                if len(waveform) == 0:
                    continue

                # Track first chunk latency
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    ttfb = (first_chunk_time - start_time) * 1000
                    logger.debug(f"TTS first chunk latency: {ttfb:.2f}ms")

                # Resample if needed
                if self._needs_resample:
                    waveform = self._resample(
                        waveform,
                        self.config.sample_rate,
                        self.config.output_sample_rate,
                    )

                # Convert to PCM16
                audio_bytes = self._float_to_pcm16(waveform)

                total_samples += len(waveform)
                chunk_count += 1

                yield AudioChunk(data=audio_bytes, is_final=False)

            # Final chunk marker
            if not self._interrupted:
                yield AudioChunk(data=b"", is_final=True)

            # Log stats
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            duration_ms = (
                total_samples / self.config.output_sample_rate * 1000
            )
            rtf = elapsed_ms / duration_ms if duration_ms > 0 else 0

            logger.info(
                f"TTS complete: {chunk_count} chunks, "
                f"{total_samples} samples ({duration_ms:.0f}ms audio), "
                f"{elapsed_ms:.0f}ms elapsed, RTF={rtf:.2f}"
            )

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise

        finally:
            self._is_speaking = False

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis (for barge-in)."""
        logger.debug("TTS interrupt requested")
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
