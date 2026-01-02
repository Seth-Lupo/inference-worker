"""
AgentRail - Coordinates VAD, ASR, LLM, and TTS rails for a voice session.
"""
import asyncio
import logging
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
        # Run VAD on chunk
        vad_event = self.vad.process_chunk(audio_chunk)

        if vad_event:
            logger.debug(f"VAD event: {vad_event.state.value} (confidence: {vad_event.confidence:.2f})")

            if vad_event.state == VADState.SPEAKING:
                # User started speaking
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
                yield AgentEvent(type="state", data={"user_speaking": False})

                # Process the accumulated audio
                if self._audio_buffer:
                    async for event in self._process_utterance():
                        yield event

        # Accumulate audio while speaking
        if self.vad.state == VADState.SPEAKING:
            self._audio_buffer += audio_chunk

    async def _process_utterance(self) -> AsyncIterator[AgentEvent]:
        """Process a complete user utterance through ASR -> LLM -> TTS."""
        self._is_processing = True
        yield AgentEvent(type="state", data={"processing": True})

        try:
            # ASR: Transcribe audio
            user_text = ""
            async for transcript in self.asr.transcribe(self._audio_buffer):
                user_text = transcript.text
                yield AgentEvent(type="transcript", data={
                    "text": transcript.text,
                    "is_final": transcript.is_final
                })

            logger.info(f"User said: {user_text}")

            # LLM: Generate response
            agent_text = ""
            async for llm_event in self.llm.generate(user_text, self._conversation_history):
                agent_text += llm_event.text
                yield AgentEvent(type="agent_text", data={
                    "text": llm_event.text,
                    "is_complete": llm_event.is_complete
                })

            logger.info(f"Agent response: {agent_text}")

            # Update conversation history
            self._conversation_history.append({"role": "user", "content": user_text})
            self._conversation_history.append({"role": "assistant", "content": agent_text})

            # TTS: Synthesize response
            yield AgentEvent(type="state", data={"agent_speaking": True})

            async for audio_chunk in self.tts.synthesize(agent_text):
                yield AgentEvent(type="audio", data=audio_chunk.data)

                # Also send via callback if set
                if self._audio_callback:
                    await self._audio_callback(audio_chunk.data)

            yield AgentEvent(type="state", data={"agent_speaking": False})

        finally:
            self._is_processing = False
            yield AgentEvent(type="state", data={"processing": False})
            self._audio_buffer = b""

    def reset(self) -> None:
        """Reset all rails for new session."""
        self.vad.reset()
        self.asr.reset()
        self.llm.reset()
        self.tts.reset()
        self._audio_buffer = b""
        self._conversation_history = []
        self._is_processing = False

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
