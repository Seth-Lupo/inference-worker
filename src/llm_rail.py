"""
LLM Rail - Mock implementation that returns fixed text.
"""
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional


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
        self._interrupted = False

        # Simulate streaming token generation
        response = "This is LLM test"

        # Stream word by word
        words = response.split()
        for i, word in enumerate(words):
            if self._interrupted:
                return

            await asyncio.sleep(0.02)  # Simulate token generation time

            is_last = i == len(words) - 1
            yield LLMEvent(
                text=word + ("" if is_last else " "),
                is_complete=is_last
            )

    def interrupt(self) -> None:
        """Interrupt ongoing generation (for barge-in)."""
        self._interrupted = True

    def reset(self) -> None:
        """Reset internal state."""
        self._interrupted = False
