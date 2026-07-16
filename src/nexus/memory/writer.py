"""Background memory writer (Phase 2).

Memory extraction used to run inside the graph, adding a full LLM round-trip to
every chat response. It now runs as a background task after the response is
sent. Failures are logged, never raised — a bad extraction must not take
anything down.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nexus.embeddings.base import EmbeddingProvider
from nexus.llm.base import LLMProvider
from nexus.memory.extractor import extract_memories
from nexus.memory.store import add_memories

logger = logging.getLogger("nexus.memory")


async def memorize_turn(
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_message: str,
    answer: str,
    source: str | None,
    owner: str = "anonymous",
) -> None:
    try:
        extracted = await extract_memories(llm, user_message=user_message, assistant_reply=answer)
        if not extracted:
            return
        async with session_factory() as session:
            await add_memories(session, embedder, extracted, owner=owner, source=source)
        logger.info("stored %d memories (source=%s)", len(extracted), source)
    except Exception:
        logger.exception("memory extraction failed (source=%s)", source)
