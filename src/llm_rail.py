"""
LLM Rail - Mock implementation that returns fixed text.
"""
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMEvent:
    text: str
    is_complete: bool


class LLMRail:
    """
    Mock LLM Rail that returns "This is LLM test" for any input.

    In production, this would use Qwen 3 8B via Triton TensorRT-LLM.
    """

    def __init__(self):
        self._interrupted = False
        logger.debug("LLMRail initialized")

    async def generate(
        self,
        user_text: str,
        conversation_history: Optional[list] = None,
    ) -> AsyncIterator[LLMEvent]:
        """
        Generate response for user text.

        Args:
            user_text: Transcribed user speech
            conversation_history: Previous conversation turns

        Yields:
            LLMEvent with generated text
        """
        start_time = time.perf_counter()
        self._interrupted = False

        logger.debug(f"LLM generate called with: '{user_text}'")

        # Mock response - no delay for speed
        response = "This is LLM test"

        # Emit full response at once for minimal latency
        logger.debug(f"LLM emitting response: '{response}'")
        yield LLMEvent(text=response, is_complete=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"LLM generation complete in {elapsed_ms:.2f}ms")

    def interrupt(self) -> None:
        """Interrupt ongoing generation (for barge-in)."""
        logger.debug("LLM interrupted")
        self._interrupted = True

    def reset(self) -> None:
        """Reset internal state."""
        logger.debug("LLM reset")
        self._interrupted = False
