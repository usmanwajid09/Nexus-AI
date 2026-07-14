from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.db.models import Memory, MemoryType
from nexus.embeddings.base import EmbeddingProvider
from nexus.memory.parsing import ExtractedMemory


async def add_memories(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    memories: list[ExtractedMemory],
    *,
    source: str | None = None,
) -> list[Memory]:
    if not memories:
        return []
    embeddings = await embedder.embed_documents([m.content for m in memories])
    rows = [
        Memory(
            type=MemoryType(m.type),
            content=m.content,
            source=source,
            embedding=embedding,
        )
        for m, embedding in zip(memories, embeddings)
    ]
    session.add_all(rows)
    await session.commit()
    return rows


async def recall(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    query: str,
    *,
    limit: int = 5,
) -> list[Memory]:
    """Semantic recall over stored memories, updating access stats.

    Access stats (count + recency) are the substrate for memory decay and
    reinforcement in a later phase.
    """
    query_embedding = await embedder.embed_query(query)
    stmt = select(Memory).order_by(Memory.embedding.cosine_distance(query_embedding)).limit(limit)
    memories = list((await session.scalars(stmt)).all())

    now = datetime.now(timezone.utc)
    for memory in memories:
        memory.last_accessed_at = now
        memory.access_count += 1
    await session.commit()
    return memories
