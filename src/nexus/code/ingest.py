"""Repository ingestion (Phase 4): walk a local checkout, chunk code at
definition boundaries, and store it in the knowledge base with kind="code" so
the code agent can retrieve it separately from prose documents."""

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from nexus.code.chunker import LANGUAGE_BY_EXTENSION, chunk_code
from nexus.db.models import Chunk, Document
from nexus.embeddings.base import EmbeddingProvider

logger = logging.getLogger("nexus.code")

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode", "target",
}


@dataclass
class RepoIngestStats:
    files_ingested: int
    files_skipped: int
    chunks: int


def _iter_source_files(root: Path, *, max_files: int) -> tuple[list[Path], int]:
    selected: list[Path] = []
    skipped = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in LANGUAGE_BY_EXTENSION:
            skipped += 1
            continue
        if len(selected) >= max_files:
            skipped += 1
            continue
        selected.append(path)
    return selected, skipped


async def ingest_repo(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    root: Path,
    *,
    max_files: int = 2000,
    max_file_bytes: int = 200_000,
) -> RepoIngestStats:
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")

    files, skipped = _iter_source_files(root, max_files=max_files)
    ingested = 0
    total_chunks = 0

    for path in files:
        try:
            if path.stat().st_size > max_file_bytes:
                skipped += 1
                continue
            source = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue

        language = LANGUAGE_BY_EXTENSION[path.suffix.lower()]
        chunks = chunk_code(source, language=language)
        if not chunks:
            skipped += 1
            continue

        embeddings = await embedder.embed_documents(chunks)
        relative = path.relative_to(root).as_posix()
        document = Document(title=relative, source=f"repo:{root}", kind="code")
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
        ingested += 1
        total_chunks += len(chunks)

    await session.commit()
    logger.info("repo ingest %s: %d files, %d chunks, %d skipped", root, ingested, total_chunks, skipped)
    return RepoIngestStats(files_ingested=ingested, files_skipped=skipped, chunks=total_chunks)
