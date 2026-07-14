import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from nexus.db.models import Chunk, Document
from nexus.embeddings.base import EmbeddingProvider
from nexus.rag.chunker import chunk_text


async def ingest_document(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    *,
    title: str,
    text: str,
    source: str | None = None,
) -> tuple[uuid.UUID, int]:
    """Chunk, embed, and store a document. Returns (document_id, chunk_count)."""
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("document is empty after chunking")

    embeddings = await embedder.embed_documents(chunks)

    document = Document(title=title, source=source)
    session.add(document)
    await session.flush()

    for position, (content, embedding) in enumerate(zip(chunks, embeddings)):
        session.add(
            Chunk(
                document_id=document.id,
                position=position,
                content=content,
                embedding=embedding,
            )
        )
    await session.commit()
    return document.id, len(chunks)
