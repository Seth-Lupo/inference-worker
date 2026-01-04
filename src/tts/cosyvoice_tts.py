"""
CosyVoice 2 TTS via Triton Inference Server.

Streaming text-to-speech with interrupt support for barge-in.

Model inputs:
- target_text: Text to synthesize (required)
- reference_wav: Reference audio for voice cloning (optional)
- reference_wav_len: Length of reference audio (optional)
- reference_text: Text spoken in reference audio (optional)

Model outputs:
- waveform: Float32 audio samples at 22050Hz
"""
import asyncio
import logging
import time
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import numpy as np

from ..triton_client import TritonClient, TritonConfig

logger = logging.getLogger(__name__)


def load_reference_audio(path: str) -> tuple[np.ndarray, int]:
    """Load reference audio file as float32 samples at 16kHz."""
    import wave
    with wave.open(path, 'rb') as wf:
        assert wf.getnchannels() == 1, "Reference audio must be mono"
        assert wf.getsampwidth() == 2, "Reference audio must be 16-bit"
        assert wf.getframerate() == 16000, "Reference audio must be 16kHz"
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, len(samples)


@dataclass
class AudioChunk:
    """Audio chunk from TTS synthesis."""
    data: bytes      # PCM16 audio bytes
    is_final: bool   # True if last chunk


@dataclass
class TTSConfig:
    """CosyVoice TTS configuration."""
    model_name: str = "cosyvoice2"
    sample_rate: int = 22050      # Native CosyVoice output rate
    output_rate: int = 16000      # Target output rate (resampled)
    streaming: bool = True        # Use streaming inference
    # Voice cloning settings
    reference_audio_path: Optional[str] = None  # Path to reference .wav file
    reference_text: Optional[str] = None        # Text spoken in reference audio


class CosyVoiceTTS:
    """
    CosyVoice 2 TTS with streaming output and interrupt support.

    Usage:
        tts = CosyVoiceTTS(client)
        async for chunk in tts.synthesize("Hello world"):
            play(chunk.data)

        # For barge-in:
        tts.interrupt()
    """

    def __init__(
        self,
        client: TritonClient,
        config: Optional[TTSConfig] = None,
    ):
        self._client = client
        self._config = config or TTSConfig()

        # State
        self._cancel_event = asyncio.Event()
        self._is_speaking = False

        # Resampling ratio
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
                    f"CosyVoice: Loaded reference audio from {self._config.reference_audio_path} "
                    f"({self._reference_wav_len} samples, {self._reference_wav_len/16000:.2f}s)"
                )
            else:
                logger.warning(f"CosyVoice: Reference audio not found: {self._config.reference_audio_path}")

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently generating audio."""
        return self._is_speaking

    def interrupt(self) -> None:
        """Interrupt ongoing synthesis (for barge-in)."""
        if self._is_speaking:
            logger.debug("TTS interrupted")
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
            logger.warning("CosyVoice: Empty text, skipping")
            return

        start = time.perf_counter()
        self._cancel_event.clear()
        self._is_speaking = True

        logger.info(f"CosyVoice: Synthesizing '{text[:60]}{'...' if len(text) > 60 else ''}' ({len(text)} chars)")

        try:
            # Build inputs for CosyVoice2 BLS model
            # Note: Streaming is controlled by server's decoupled mode, not a client input
            inputs = {
                "target_text": np.array([[text]], dtype=object),
            }

            # Add reference audio for voice cloning if configured
            if self._reference_wav is not None and self._config.reference_text:
                inputs["reference_wav"] = self._reference_wav.reshape(1, -1)
                inputs["reference_wav_len"] = np.array([[self._reference_wav_len]], dtype=np.int32)
                inputs["reference_text"] = np.array([[self._config.reference_text]], dtype=object)
                logger.debug(f"CosyVoice: Using voice cloning with {self._reference_wav_len} samples")

            logger.debug(f"CosyVoice: Inputs - {list(inputs.keys())}")

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
                    logger.info("CosyVoice: Cancelled mid-stream")
                    break

                # Extract waveform
                waveform = result.outputs.get("waveform")
                if waveform is None:
                    logger.debug(f"CosyVoice: No waveform in result, outputs={list(result.outputs.keys())}")
                    continue

                waveform = waveform.flatten().astype(np.float32)
                if len(waveform) == 0:
                    logger.debug("CosyVoice: Empty waveform chunk")
                    continue

                # Log first chunk latency (TTFB)
                if first_chunk:
                    ttfb = (time.perf_counter() - start) * 1000
                    logger.info(f"CosyVoice: TTFB={ttfb:.0f}ms, chunk_samples={len(waveform)}")
                    first_chunk = False

                # Resample 22050Hz -> 16000Hz
                if self._resample_ratio != 1.0:
                    original_len = len(waveform)
                    waveform = self._resample(waveform)
                    logger.debug(f"CosyVoice: Resampled {original_len} -> {len(waveform)} samples")

                # Convert to PCM16
                audio = self._to_pcm16(waveform)

                total_samples += len(waveform)
                chunk_count += 1

                logger.debug(f"CosyVoice: Chunk {chunk_count}: {len(audio)} bytes ({len(waveform)} samples)")
                yield AudioChunk(data=audio, is_final=False)

            # Final chunk marker (if not interrupted)
            if not self._cancel_event.is_set():
                yield AudioChunk(data=b"", is_final=True)

                # Stats
                elapsed = (time.perf_counter() - start) * 1000
                duration_ms = total_samples / self._config.output_rate * 1000 if total_samples > 0 else 0
                rtf = elapsed / duration_ms if duration_ms > 0 else 0
                logger.info(
                    f"CosyVoice: Complete - {chunk_count} chunks, "
                    f"{total_samples} samples ({duration_ms:.0f}ms audio) in {elapsed:.0f}ms (RTF={rtf:.2f})"
                )

        except Exception as e:
            logger.error(f"CosyVoice: Synthesis error: {e}", exc_info=True)
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


async def create_cosyvoice_tts(
    host: str = "localhost",
    port: int = 8001,
    model_name: str = "cosyvoice2",
) -> CosyVoiceTTS:
    """
    Factory to create connected CosyVoice TTS.

    Args:
        host: Triton server host
        port: Triton gRPC port
        model_name: CosyVoice model name

    Returns:
        Connected CosyVoiceTTS instance
    """
    config = TritonConfig(host=host, port=port)
    client = TritonClient(config)
    await client.connect()

    tts_config = TTSConfig(model_name=model_name)
    return CosyVoiceTTS(client, tts_config)
