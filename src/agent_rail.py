"""
AgentRail - Coordinates VAD, ASR, LLM, and TTS rails for a voice session.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional, Awaitable

from .vad import SileroVAD, VADState, VADEvent
from .asr_rail import ASRRail, TranscriptEvent
from .llm_rail import LLMRail, LLMEvent
from .tts_rail import TTSRail, AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """Event emitted by the agent pipeline."""
    type: str  # "transcript", "agent_text", "audio", "state"
    data: any


class AgentRail:
    """
    Per-session voice agent pipeline.

    Coordinates:
    - VAD: Detects when user starts/stops speaking
    - ASR: Transcribes user speech to text
    - LLM: Generates agent response
    - TTS: Synthesizes response to audio

    Flow:
    1. Audio comes in continuously
    2. VAD detects speech start/end
    3. On speech end, ASR transcribes accumulated audio
    4. LLM generates response
    5. TTS synthesizes and streams audio back
    6. If user speaks during TTS (barge-in), interrupt and listen
    """

    def __init__(
        self,
        vad: Optional[SileroVAD] = None,
        asr: Optional[ASRRail] = None,
        llm: Optional[LLMRail] = None,
        tts: Optional[TTSRail] = None,
    ):
        """
        Initialize AgentRail with dependency injection.

        Args:
            vad: Voice activity detector (creates default if None)
            asr: ASR rail (creates default if None)
            llm: LLM rail (creates default if None)
            tts: TTS rail (creates default if None)
        """
        logger.debug("Initializing AgentRail...")
        init_start = time.perf_counter()

        self.vad = vad or SileroVAD()
        self.asr = asr or ASRRail()
        self.llm = llm or LLMRail()
        self.tts = tts or TTSRail()

        # Audio buffer for accumulating speech
        self._audio_buffer: bytes = b""

        # State
        self._is_processing = False
        self._conversation_history: list = []

        # Callback for sending audio back
        self._audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None

        # Output queue for events
        self._output_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

        # Background task for TTS
        self._tts_task: Optional[asyncio.Task] = None

        elapsed_ms = (time.perf_counter() - init_start) * 1000
        logger.info(f"AgentRail initialized in {elapsed_ms:.1f}ms")

    def set_audio_callback(self, callback: Callable[[bytes], Awaitable[None]]) -> None:
        """Set callback for sending audio back to client."""
        self._audio_callback = callback

    async def process_audio(self, audio_chunk: bytes) -> AsyncIterator[AgentEvent]:
        """
        Process incoming audio chunk.

        Args:
            audio_chunk: PCM16 audio (16kHz, mono)

        Yields:
            AgentEvent for state changes, transcripts, and audio
        """
        chunk_size = len(audio_chunk)
        logger.debug(f"Processing audio chunk: {chunk_size} bytes")

        # Run VAD on chunk
        vad_event = self.vad.process_chunk(audio_chunk)

        if vad_event:
            logger.debug(f"VAD event received: {vad_event.state.value} (confidence: {vad_event.confidence:.2f})")

            if vad_event.state == VADState.SPEAKING:
                # User started speaking
                logger.info("User started speaking")
                yield AgentEvent(type="state", data={"user_speaking": True})

                # If TTS is playing, interrupt it (barge-in)
                if self.tts.is_speaking:
                    logger.info("Barge-in detected, interrupting TTS")
                    self.tts.interrupt()
                    self.llm.interrupt()
                    if self._tts_task:
                        self._tts_task.cancel()
                        try:
                            await self._tts_task
                        except asyncio.CancelledError:
                            pass

                # Start accumulating audio
                self._audio_buffer = b""

            elif vad_event.state == VADState.END_OF_SPEECH:
                # User stopped speaking
                buffer_size = len(self._audio_buffer)
                buffer_duration_ms = (buffer_size / 2) / 16  # PCM16 = 2 bytes/sample, 16kHz
                logger.info(f"User stopped speaking. Buffer: {buffer_size} bytes ({buffer_duration_ms:.0f}ms)")
                yield AgentEvent(type="state", data={"user_speaking": False})

                # Process the accumulated audio
                if self._audio_buffer:
                    logger.debug("Starting utterance processing pipeline")
                    async for event in self._process_utterance():
                        yield event
                else:
                    logger.warning("Empty audio buffer, skipping processing")

        # Accumulate audio while speaking
        if self.vad.state == VADState.SPEAKING:
            self._audio_buffer += audio_chunk

    async def _process_utterance(self) -> AsyncIterator[AgentEvent]:
        """Process a complete user utterance through ASR -> LLM -> TTS."""
        pipeline_start = time.perf_counter()
        self._is_processing = True
        yield AgentEvent(type="state", data={"processing": True})

        try:
            # ASR: Transcribe audio
            logger.debug("Starting ASR transcription...")
            asr_start = time.perf_counter()
            user_text = ""
            async for transcript in self.asr.transcribe(self._audio_buffer):
                user_text = transcript.text
                yield AgentEvent(type="transcript", data={
                    "text": transcript.text,
                    "is_final": transcript.is_final
                })
            asr_elapsed = (time.perf_counter() - asr_start) * 1000
            logger.info(f"ASR complete in {asr_elapsed:.2f}ms: '{user_text}'")

            # LLM: Generate response
            logger.debug("Starting LLM generation...")
            llm_start = time.perf_counter()
            agent_text = ""
            async for llm_event in self.llm.generate(user_text, self._conversation_history):
                agent_text += llm_event.text
                yield AgentEvent(type="agent_text", data={
                    "text": llm_event.text,
                    "is_complete": llm_event.is_complete
                })
            llm_elapsed = (time.perf_counter() - llm_start) * 1000
            logger.info(f"LLM complete in {llm_elapsed:.2f}ms: '{agent_text}'")

            # Update conversation history
            self._conversation_history.append({"role": "user", "content": user_text})
            self._conversation_history.append({"role": "assistant", "content": agent_text})

            # TTS: Synthesize response
            logger.debug("Starting TTS synthesis...")
            tts_start = time.perf_counter()
            yield AgentEvent(type="state", data={"agent_speaking": True})

            audio_bytes_sent = 0
            async for audio_chunk in self.tts.synthesize(agent_text):
                audio_bytes_sent += len(audio_chunk.data)
                yield AgentEvent(type="audio", data=audio_chunk.data)

                # Also send via callback if set
                if self._audio_callback:
                    await self._audio_callback(audio_chunk.data)

            tts_elapsed = (time.perf_counter() - tts_start) * 1000
            logger.info(f"TTS complete in {tts_elapsed:.2f}ms: {audio_bytes_sent} bytes sent")

            yield AgentEvent(type="state", data={"agent_speaking": False})

            # Log total pipeline time
            total_elapsed = (time.perf_counter() - pipeline_start) * 1000
            logger.info(f"Pipeline complete: ASR={asr_elapsed:.0f}ms + LLM={llm_elapsed:.0f}ms + TTS={tts_elapsed:.0f}ms = {total_elapsed:.0f}ms total")

        finally:
            self._is_processing = False
            yield AgentEvent(type="state", data={"processing": False})
            self._audio_buffer = b""

    def reset(self) -> None:
        """Reset all rails for new session."""
        logger.debug("Resetting AgentRail")
        self.vad.reset()
        self.asr.reset()
        self.llm.reset()
        self.tts.reset()
        self._audio_buffer = b""
        self._conversation_history = []
        self._is_processing = False
        logger.info("AgentRail reset complete")

    @property
    def is_processing(self) -> bool:
        """Whether the pipeline is currently processing."""
        return self._is_processing

    @property
    def is_user_speaking(self) -> bool:
        """Whether the user is currently speaking."""
        return self.vad.is_speaking

    @property
    def is_agent_speaking(self) -> bool:
        """Whether the agent is currently speaking."""
        return self.tts.is_speaking
