"""
Chatterbox Turbo TTS via Triton Inference Server.

Streaming text-to-speech with GPU acceleration.
Supports voice cloning via reference audio.

Model inputs:
- target_text: Text to synthesize (required)
- reference_wav: Reference audio for voice cloning (optional)
- reference_wav_len: Length of reference audio (optional)
- exaggeration: Emotion exaggeration level 0.0-1.0 (optional)

Model outputs:
- waveform: Float32 audio samples at 24kHz
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np

from ..triton_client import TritonClient

logger = logging.getLogger(__name__)


def load_reference_audio(path: str, target_sr: int = 16000) -> tuple:
    """Load reference audio file as float32 samples."""
    import wave

    with wave.open(path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    # Convert to float32
    if sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Convert stereo to mono
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    # Resample if needed
    if framerate != target_sr:
        # Simple linear interpolation resampling
        original_length = len(samples)
        new_length = int(original_length * target_sr / framerate)
        indices = np.linspace(0, original_length - 1, new_length)
        samples = np.interp(indices, np.arange(original_length), samples).astype(np.float32)

    return samples, len(samples)


@dataclass
class AudioChunk:
    """Audio chunk from TTS synthesis."""
    data: bytes      # PCM16 audio bytes
    is_final: bool   # True if last chunk


@dataclass
class ChatterboxConfig:
    """Chatterbox TTS configuration."""
    model_name: str = "chatterbox"
    sample_rate: int = 24000       # Native Chatterbox output rate
    output_rate: int = 16000       # Target output rate (resampled for telephony)
    streaming: bool = True
    exaggeration: float = 0.5      # Emotion exaggeration (0=monotone, 1=dramatic)
    # Voice cloning settings
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None


class ChatterboxTTS:
    """
    Chatterbox Turbo TTS with streaming output and interrupt support.

    Features:
    - GPU-accelerated inference (PyTorch + torch.compile)
    - Streaming audio output for low latency
    - Voice cloning via reference audio
    - Emotion exaggeration control
    - Interrupt support for barge-in

    Usage:
        tts = ChatterboxTTS(client)
        async for chunk in tts.synthesize("Hello world"):
            play(chunk.data)

        # For barge-in:
        tts.interrupt()
    """

    def __init__(
        self,
        client: TritonClient,
        config: Optional[ChatterboxConfig] = None,
    ):
        self._client = client
        self._config = config or ChatterboxConfig()

        # State
        self._cancel_event = asyncio.Event()
        self._is_speaking = False

        # Resampling ratio for output
        self._resample_ratio = self._config.output_rate / self._config.sample_rate

        # Load reference audio for voice cloning
        self._reference_wav: Optional[np.ndarray] = None
        self._reference_wav_len: Optional[int] = None

        if self._config.reference_audio_path:
            if os.path.exists(self._config.reference_audio_path):
                self._reference_wav, self._reference_wav_len = load_reference_audio(
                    self._config.reference_audio_path
                )
                logger.info(
                    f"Chatterbox: Loaded reference audio from {self._config.reference_audio_path} "
                    f"({self._reference_wav_len} samples, {self._reference_wav_len/16000:.2f}s)"
                )
            else:
                logger.warning(
                    f"Chatterbox: Reference audio not found: {self._config.reference_audio_path}"
                )

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently generating audio."""
        return self._is_speaking

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis (for barge-in)."""
        if self._is_speaking:
            logger.debug("Chatterbox: Interrupted")
            self._cancel_event.set()

    def reset(self) -> None:
        """Reset state for new synthesis."""
        self._cancel_event.clear()
        self._is_speaking = False

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize text to streaming audio.

        Args:
            text: Text to synthesize

        Yields:
            AudioChunk with PCM16 audio data
        """
        if not text.strip():
            logger.warning("Chatterbox: Empty text, skipping")
            return

        start = time.perf_counter()
        self._cancel_event.clear()
        self._is_speaking = True

        logger.info(
            f"Chatterbox: Synthesizing '{text[:60]}{'...' if len(text) > 60 else ''}' "
            f"({len(text)} chars)"
        )

        try:
            # Build inputs
            inputs = {
                "target_text": np.array([[text]], dtype=object),
            }

            # Add reference audio for voice cloning
            if self._reference_wav is not None:
                inputs["reference_wav"] = self._reference_wav.reshape(1, -1)
                inputs["reference_wav_len"] = np.array(
                    [[self._reference_wav_len]], dtype=np.int32
                )
                logger.debug(
                    f"Chatterbox: Using voice cloning with {self._reference_wav_len} samples"
                )

            # Add exaggeration control
            inputs["exaggeration"] = np.array(
                [[self._config.exaggeration]], dtype=np.float32
            )

            logger.debug(f"Chatterbox: Inputs - {list(inputs.keys())}")

            # Stream inference
            first_chunk = True
            total_samples = 0
            chunk_count = 0

            async for result in self._client.infer_stream(
                model_name=self._config.model_name,
                inputs=inputs,
                output_names=["waveform"],
                cancel_event=self._cancel_event,
            ):
                # Check interrupt
                if self._cancel_event.is_set():
                    logger.info("Chatterbox: Cancelled mid-stream")
                    break

                # Extract waveform
                waveform = result.outputs.get("waveform")
                if waveform is None:
                    logger.debug(
                        f"Chatterbox: No waveform in result, "
                        f"outputs={list(result.outputs.keys())}"
                    )
                    continue

                waveform = waveform.flatten().astype(np.float32)
                if len(waveform) == 0:
                    logger.debug("Chatterbox: Empty waveform chunk")
                    continue

                # Log first chunk latency (TTFB)
                if first_chunk:
                    ttfb = (time.perf_counter() - start) * 1000
                    logger.info(
                        f"Chatterbox: TTFB={ttfb:.0f}ms, chunk_samples={len(waveform)}"
                    )
                    first_chunk = False

                # Resample 24kHz -> 16kHz if needed
                if self._resample_ratio != 1.0:
                    original_len = len(waveform)
                    waveform = self._resample(waveform)
                    logger.debug(
                        f"Chatterbox: Resampled {original_len} -> {len(waveform)} samples"
                    )

                # Convert to PCM16
                audio = self._to_pcm16(waveform)

                total_samples += len(waveform)
                chunk_count += 1

                logger.debug(
                    f"Chatterbox: Chunk {chunk_count}: {len(audio)} bytes "
                    f"({len(waveform)} samples)"
                )
                yield AudioChunk(data=audio, is_final=False)

            # Final chunk marker (if not interrupted)
            if not self._cancel_event.is_set():
                yield AudioChunk(data=b"", is_final=True)

                # Stats
                elapsed = (time.perf_counter() - start) * 1000
                duration_ms = total_samples / self._config.output_rate * 1000 if total_samples > 0 else 0
                rtf = elapsed / duration_ms if duration_ms > 0 else 0

                logger.info(
                    f"Chatterbox: Complete - {chunk_count} chunks, "
                    f"{total_samples} samples ({duration_ms:.0f}ms audio) in "
                    f"{elapsed:.0f}ms (RTF={rtf:.2f})"
                )

        except Exception as e:
            logger.error(f"Chatterbox: Synthesis error: {e}", exc_info=True)
            raise
        finally:
            self._is_speaking = False

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio using linear interpolation."""
        new_len = int(len(audio) * self._resample_ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _to_pcm16(self, audio: np.ndarray) -> bytes:
        """Convert float32 [-1,1] to PCM16 bytes."""
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()


async def create_chatterbox_tts(
    host: str = "localhost",
    port: int = 8001,
    model_name: str = "chatterbox",
    reference_audio_path: Optional[str] = None,
    exaggeration: float = 0.5,
) -> ChatterboxTTS:
    """
    Factory to create connected Chatterbox TTS.

    Args:
        host: Triton server host
        port: Triton gRPC port
        model_name: Chatterbox model name
        reference_audio_path: Path to reference audio for voice cloning
        exaggeration: Emotion exaggeration level (0.0-1.0)

    Returns:
        Connected ChatterboxTTS instance
    """
    from ..triton_client import TritonConfig

    config = TritonConfig(host=host, port=port)
    client = TritonClient(config)
    await client.connect()

    tts_config = ChatterboxConfig(
        model_name=model_name,
        reference_audio_path=reference_audio_path,
        exaggeration=exaggeration,
    )
    return ChatterboxTTS(client, tts_config)
