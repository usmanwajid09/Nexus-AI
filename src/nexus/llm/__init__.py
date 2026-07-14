from nexus.config import Settings
from nexus.llm.base import LLMProvider


def get_llm(settings: Settings) -> LLMProvider:
    from nexus.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(model=settings.llm_model)
