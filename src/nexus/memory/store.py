from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.config import get_settings
from nexus.db.models import Memory, MemoryType
from nexus.embeddings.base import EmbeddingProvider
from nexus.memory.decay import memory_score
from nexus.memory.parsing import ExtractedMemory


async def add_memories(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    memories: list[ExtractedMemory],
    *,
    owner: str = "anonymous",
    source: str | None = None,
) -> list[Memory]:
    if not memories:
        return []
    embeddings = await embedder.embed_documents([m.content for m in memories])
    rows = [
        Memory(
            owner=owner,
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
    owner: str = "anonymous",
    limit: int = 5,
) -> list[Memory]:
    """Decay-weighted semantic recall (Phase 6).

    Over-fetches by similarity, then re-scores candidates with a forgetting
    curve: score = similarity * decay_weight(age, accesses). Old, never-used
    memories fade; frequently recalled ones persist. Returned memories get
    their access stats bumped, which reinforces them for future recalls.
    """
    settings = get_settings()
    query_embedding = await embedder.embed_query(query)

    distance = Memory.embedding.cosine_distance(query_embedding).label("distance")
    stmt = (
        select(Memory, distance)
        .where(Memory.owner == owner)
        .order_by(distance)
        .limit(max(limit * 4, 20))
    )
    rows = (await session.execute(stmt)).all()

    now = datetime.now(timezone.utc)
    scored: list[tuple[float, Memory]] = []
    for memory, dist in rows:
        anchor = memory.last_accessed_at or memory.created_at
        age_days = max((now - anchor).total_seconds() / 86400.0, 0.0)
        score = memory_score(
            1.0 - float(dist),
            age_days,
            memory.access_count,
            half_life_days=settings.memory_half_life_days,
            reinforcement=settings.memory_reinforcement,
        )
        scored.append((score, memory))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    memories = [memory for _, memory in scored[:limit]]

    for memory in memories:
        memory.last_accessed_at = now
        memory.access_count += 1
    await session.commit()
    return memories
