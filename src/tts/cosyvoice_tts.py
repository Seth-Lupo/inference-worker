"""
CosyVoice 2 TTS via Triton Inference Server.

Streaming text-to-speech with interrupt support for barge-in.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import numpy as np

from ..triton_client import TritonClient, TritonConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Audio chunk from TTS synthesis."""
    data: bytes      # PCM16 audio bytes
    is_final: bool   # True if last chunk


@dataclass
class TTSConfig:
    """CosyVoice TTS configuration."""
    model_name: str = "cosyvoice2"
    sample_rate: int = 22050      # Native CosyVoice rate
    output_rate: int = 16000      # Output rate (resampled)


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
            return

        start = time.perf_counter()
        self._cancel_event.clear()
        self._is_speaking = True

        logger.debug(f"TTS: '{text[:50]}...' ({len(text)} chars)")

        try:
            # Build input
            inputs = {
                "target_text": np.array([[text]], dtype=object),
            }

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
                    logger.debug("TTS cancelled mid-stream")
                    break

                # Extract waveform
                waveform = result.outputs.get("waveform")
                if waveform is None or len(waveform.flatten()) == 0:
                    continue

                waveform = waveform.flatten().astype(np.float32)

                # Log first chunk latency
                if first_chunk:
                    ttfb = (time.perf_counter() - start) * 1000
                    logger.info(f"TTS first chunk: {ttfb:.0f}ms")
                    first_chunk = False

                # Resample if needed
                if self._resample_ratio != 1.0:
                    waveform = self._resample(waveform)

                # Convert to PCM16
                audio = self._to_pcm16(waveform)

                total_samples += len(waveform)
                chunk_count += 1

                yield AudioChunk(data=audio, is_final=False)

            # Final chunk (if not interrupted)
            if not self._cancel_event.is_set():
                yield AudioChunk(data=b"", is_final=True)

                # Stats
                elapsed = (time.perf_counter() - start) * 1000
                duration = total_samples / self._config.output_rate * 1000
                rtf = elapsed / duration if duration > 0 else 0
                logger.info(
                    f"TTS done: {chunk_count} chunks, "
                    f"{duration:.0f}ms audio in {elapsed:.0f}ms (RTF={rtf:.2f})"
                )

        except Exception as e:
            logger.error(f"TTS error: {e}")
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
