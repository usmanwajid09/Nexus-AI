from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMProvider(ABC):
    """Provider-agnostic LLM interface.

    Everything above this layer (orchestrator, memory extractor, agents)
    depends only on this interface, so swapping or adding providers is a
    one-file change. `complete` and `complete_json` are required; vision and
    web research are optional capabilities a provider may not have.
    """

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 16000,
    ) -> str:
        """Return the assistant's text reply for a chat-shaped request."""

    @abstractmethod
    async def complete_json(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Return a dict guaranteed to match the given JSON schema."""

    async def complete_stream(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 16000,
    ) -> AsyncIterator[str]:
        """Yield the reply as text deltas.

        Default falls back to one chunk via complete(), so every provider is
        streamable from the caller's point of view.
        """
        yield await self.complete(system=system, messages=messages, max_tokens=max_tokens)

    async def complete_vision(
        self,
        *,
        image: bytes,
        media_type: str,
        question: str,
        max_tokens: int = 4096,
    ) -> str:
        """Answer a question about an image (optional capability)."""
        raise NotImplementedError(f"{type(self).__name__} does not support vision")

    async def research(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 16000,
    ) -> str:
        """Answer using live web search/fetch tools (optional capability)."""
        raise NotImplementedError(f"{type(self).__name__} does not support web research")
