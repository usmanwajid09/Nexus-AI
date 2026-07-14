from typing import Any

from nexus.llm.base import LLMProvider


class FakeLLM(LLMProvider):
    async def complete(self, *, system: str, messages: list, max_tokens: int = 16000) -> str:
        return "full answer"

    async def complete_json(self, *, prompt: str, schema: dict, max_tokens: int = 2048) -> dict[str, Any]:
        return {}


async def test_default_stream_yields_complete_result():
    llm = FakeLLM()
    chunks = [c async for c in llm.complete_stream(system="s", messages=[])]
    assert chunks == ["full answer"]


async def test_optional_capabilities_raise_not_implemented():
    llm = FakeLLM()
    import pytest

    with pytest.raises(NotImplementedError):
        await llm.complete_vision(image=b"", media_type="image/png", question="?")
    with pytest.raises(NotImplementedError):
        await llm.research(system="s", messages=[])
