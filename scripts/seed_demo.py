"""Seed the knowledge base with demo documents and memories.

Usage:
    python scripts/seed_demo.py

The documents match evals/golden.jsonl, so a fresh setup gets a meaningful
eval baseline immediately:

    docker compose up -d
    python scripts/seed_demo.py
    python scripts/run_evals.py

Idempotent: re-running replaces the previously seeded data (matched on
source="seed:demo"). Works without any API keys when EMBEDDING_PROVIDER=hashing.
"""

import asyncio

from sqlalchemy import delete

from nexus.config import get_settings
from nexus.db.models import Document, Memory
from nexus.db.session import get_session_factory, init_db
from nexus.embeddings import get_embedder
from nexus.memory.parsing import ExtractedMemory
from nexus.memory.store import add_memories
from nexus.rag.ingest import ingest_document

SEED_SOURCE = "seed:demo"

DOCUMENTS = [
    {
        "title": "Architecture notes",
        "text": (
            "Our backend is built on FastAPI with async SQLAlchemy. The API gateway "
            "terminates TLS and handles authentication using JWT bearer tokens signed "
            "with HS256; tokens are minted by the auth service and expire after one hour.\n\n"
            "All persistent state lives in PostgreSQL. Vector search uses the pgvector "
            "extension; full-text search uses Postgres tsvector. We deliberately avoid "
            "additional datastores until a measured limit forces one.\n\n"
            "Services communicate over plain HTTP inside the private network. The "
            "orchestration layer is a LangGraph state machine that routes requests to "
            "specialist pipelines."
        ),
    },
    {
        "title": "Deployment runbook",
        "text": (
            "Deployments go through make targets. Deploy to staging with `make deploy "
            "ENV=staging`; production requires a tagged release and `make deploy "
            "ENV=prod`. Every deploy runs the migration check first and aborts if a "
            "migration is pending.\n\n"
            "To roll back a bad release, run `make rollback ENV=prod` which redeploys "
            "the previous tagged image and restores the config snapshot taken at deploy "
            "time. Rollbacks do not revert database migrations - those need a forward "
            "fix.\n\n"
            "Deploy windows are Monday to Thursday before 16:00. Friday deploys need "
            "an explicit sign-off from the on-call engineer."
        ),
    },
    {
        "title": "Database guide",
        "text": (
            "The primary database is PostgreSQL 17 with the pgvector extension. "
            "Backups run nightly at 02:00 UTC via pg_dump to the offsite bucket, with "
            "point-in-time recovery enabled through WAL archiving. Restore drills run "
            "on the first Monday of each month.\n\n"
            "Connection pooling is handled by the application (SQLAlchemy async pool, "
            "max 20 connections per instance). Long-running analytical queries must go "
            "to the read replica, never the primary."
        ),
    },
    {
        "title": "Team onboarding",
        "text": (
            "Code style: Python is formatted with ruff; line length 100. Every change "
            "ships through a pull request - no direct pushes to main. A pull request "
            "needs one approving review and a green CI run before merge; squash-merge "
            "is the default.\n\n"
            "New joiners pair with their onboarding buddy for the first two weeks. The "
            "first-week task is always a small, real bug fix that touches the full "
            "deploy pipeline end to end."
        ),
    },
    {
        "title": "Incident postmortem 2026-03",
        "text": (
            "On 2026-03-12 the API was degraded for 47 minutes. Root cause: connection "
            "pool exhaustion on the primary database after a batch job was accidentally "
            "pointed at the primary instead of the read replica, starving the API of "
            "connections.\n\n"
            "Fixes shipped: batch jobs now use a dedicated credential that can only "
            "reach the replica, pool saturation alerts at 80%, and the runbook gained "
            "a connection-pool triage section."
        ),
    },
]

MEMORIES = [
    ExtractedMemory(type="semantic", content="The backend uses FastAPI with async SQLAlchemy."),
    ExtractedMemory(type="procedural", content="Deploy to staging with `make deploy ENV=staging`."),
    ExtractedMemory(
        type="episodic",
        content="In March 2026 an incident was caused by database connection pool exhaustion.",
    ),
]


async def main() -> None:
    settings = get_settings()
    embedder = get_embedder(settings)
    await init_db()
    session_factory = get_session_factory()

    async with session_factory() as session:
        await session.execute(delete(Document).where(Document.source == SEED_SOURCE))
        await session.execute(delete(Memory).where(Memory.source == SEED_SOURCE))
        await session.commit()

    total_chunks = 0
    async with session_factory() as session:
        for doc in DOCUMENTS:
            _, chunks = await ingest_document(
                session, embedder, title=doc["title"], text=doc["text"], source=SEED_SOURCE
            )
            total_chunks += chunks
        await add_memories(session, embedder, MEMORIES, source=SEED_SOURCE)

    print(f"seeded {len(DOCUMENTS)} documents ({total_chunks} chunks) and {len(MEMORIES)} memories")
    print("next: python scripts/run_evals.py")


if __name__ == "__main__":
    asyncio.run(main())
