import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.db.models import Chunk, Document
from nexus.embeddings.base import EmbeddingProvider
from nexus.rag.fusion import rrf_fuse
from nexus.rag.rerank import Reranker


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    content: str
    document_title: str


def _with_kind(stmt: Select, kind: str | None) -> Select:
    if kind is not None:
        stmt = stmt.join(Document, Chunk.document_id == Document.id).where(Document.kind == kind)
    return stmt


async def vector_search(
    session: AsyncSession, embedding: list[float], limit: int, *, kind: str | None = None
) -> list[uuid.UUID]:
    stmt = _with_kind(
        select(Chunk.id).order_by(Chunk.embedding.cosine_distance(embedding)).limit(limit), kind
    )
    return list((await session.scalars(stmt)).all())


async def keyword_search(
    session: AsyncSession, query: str, limit: int, *, kind: str | None = None
) -> list[uuid.UUID]:
    tsquery = func.plainto_tsquery("english", query)
    tsvector = func.to_tsvector("english", Chunk.content)
    stmt = _with_kind(
        select(Chunk.id)
        .where(tsvector.op("@@")(tsquery))
        .order_by(func.ts_rank(tsvector, tsquery).desc())
        .limit(limit),
        kind,
    )
    return list((await session.scalars(stmt)).all())


async def retrieve(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    queries: Sequence[str],
    *,
    limit: int = 6,
    kind: str | None = None,
    reranker: Reranker | None = None,
    rerank_query: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid multi-query retrieval.

    Every query contributes a vector ranking and a keyword ranking; all
    rankings are fused with RRF. When a reranker is given, fusion over-fetches
    2x and the reranker picks the final order.
    """
    if not queries:
        return []

    rankings: list[list[uuid.UUID]] = []
    for query in queries:
        embedding = await embedder.embed_query(query)
        rankings.append(await vector_search(session, embedding, limit * 2, kind=kind))
        rankings.append(await keyword_search(session, query, limit * 2, kind=kind))

    candidate_count = limit * 2 if reranker is not None else limit
    fused_ids = rrf_fuse(rankings)[:candidate_count]
    if not fused_ids:
        return []

    stmt = (
        select(Chunk.id, Chunk.content, Document.title)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(fused_ids))
    )
    rows = {row.id: row for row in (await session.execute(stmt)).all()}
    chunks = [
        RetrievedChunk(chunk_id=cid, content=rows[cid].content, document_title=rows[cid].title)
        for cid in fused_ids
        if cid in rows
    ]

    if reranker is not None and chunks:
        order = await reranker.rerank(rerank_query or queries[0], [c.content for c in chunks])
        chunks = [chunks[i] for i in order]
    return chunks[:limit]
