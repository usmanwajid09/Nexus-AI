import base64
import json
from typing import Any

import anthropic

from nexus.llm.base import LLMProvider
from nexus.observability import record_llm_usage

REFUSAL_FALLBACK = (
    "I can't help with that request. If you think this is a mistake, "
    "try rephrasing or narrowing the question."
)

# Server-side tools: Anthropic runs search/fetch, no client-side loop needed.
_WEB_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 5},
]


def _text_of(response: anthropic.types.Message) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text")


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-opus-4-8", *, max_continuations: int = 5) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._max_continuations = max_continuations

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 16000,
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
        )
        record_llm_usage("complete", response.usage)
        if response.stop_reason == "refusal":
            return REFUSAL_FALLBACK
        return next((b.text for b in response.content if b.type == "text"), "")

    async def complete_stream(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 16000,
    ):
        streamed_any = False
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    streamed_any = True
                    yield text
            final = await stream.get_final_message()
        record_llm_usage("complete", final.usage)
        if final.stop_reason == "refusal" and not streamed_any:
            yield REFUSAL_FALLBACK

    async def complete_json(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        record_llm_usage("complete_json", response.usage)
        if response.stop_reason == "refusal":
            return {}
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        return json.loads(text)

    async def complete_vision(
        self,
        *,
        image: bytes,
        media_type: str,
        question: str,
        max_tokens: int = 4096,
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.standard_b64encode(image).decode(),
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        record_llm_usage("vision", response.usage)
        if response.stop_reason == "refusal":
            return REFUSAL_FALLBACK
        return _text_of(response)

    async def research(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 16000,
    ) -> str:
        """Web-grounded answer via server-side search/fetch tools.

        The server runs its own tool loop; if it pauses (`pause_turn`), we
        append the partial assistant turn and re-send so it resumes.
        """
        conversation: list[dict[str, Any]] = list(messages)
        response = None
        for _ in range(self._max_continuations):
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=system,
                tools=_WEB_TOOLS,
                messages=conversation,
            )
            record_llm_usage("research", response.usage)
            if response.stop_reason == "refusal":
                return REFUSAL_FALLBACK
            if response.stop_reason == "pause_turn":
                conversation = [*conversation, {"role": "assistant", "content": response.content}]
                continue
            return _text_of(response)
        return _text_of(response) if response is not None else ""
