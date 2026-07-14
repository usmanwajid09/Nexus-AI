import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select

from nexus.api.auth import require_auth
from nexus.api.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    MemoryOut,
    RepoIngestRequest,
    RepoIngestResponse,
    VisionResponse,
)
from nexus.code.ingest import ingest_repo
from nexus.config import get_settings
from nexus.db.models import Conversation, Message
from nexus.db.session import get_session_factory, init_db
from nexus.embeddings import get_embedder
from nexus.llm import get_llm
from nexus.memory.store import recall
from nexus.memory.writer import memorize_turn
from nexus.observability import REQUEST_LATENCY, metrics_app
from nexus.orchestrator.graph import build_graph
from nexus.rag.ingest import ingest_document

logger = logging.getLogger("nexus")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    app.state.settings = settings
    app.state.llm = get_llm(settings)
    app.state.embedder = get_embedder(settings)
    app.state.graph = build_graph(
        llm=app.state.llm,
        embedder=app.state.embedder,
        session_factory=get_session_factory(),
        settings=settings,
    )
    if settings.auth_secret and len(settings.auth_secret) < 32:
        logger.warning("AUTH_SECRET is under 32 characters - use a longer random secret")
    logger.info(
        "nexus started (model=%s, embeddings=%s, auth=%s)",
        settings.llm_model,
        settings.embedding_provider,
        "on" if settings.auth_secret else "off (dev mode)",
    )
    yield


app = FastAPI(title="Nexus AI", version="0.5.0", lifespan=lifespan)
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    REQUEST_LATENCY.labels(
        method=request.method, path=request.url.path, status=response.status_code
    ).observe(elapsed)
    logger.info(
        "req_id=%s %s %s -> %s in %.0fms",
        request_id, request.method, request.url.path, response.status_code, elapsed * 1000,
    )
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user: str = Depends(require_auth),
) -> ChatResponse:
    settings = request.app.state.settings
    session_factory = get_session_factory()

    async with session_factory() as session:
        if req.conversation_id is not None:
            conversation = await session.get(Conversation, req.conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="conversation not found")
        else:
            conversation = Conversation(title=req.message[:80])
            session.add(conversation)
            await session.flush()

        history_rows = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc())
                .limit(settings.max_history_messages)
            )
        ).all()
        history = [{"role": m.role, "content": m.content} for m in reversed(history_rows)]

        session.add(Message(conversation_id=conversation.id, role="user", content=req.message))
        await session.commit()
        conversation_id = conversation.id

    result = await request.app.state.graph.ainvoke(
        {
            "conversation_id": str(conversation_id),
            "user_message": req.message,
            "history": history,
        }
    )

    async with session_factory() as session:
        session.add(
            Message(conversation_id=conversation_id, role="assistant", content=result["answer"])
        )
        await session.commit()

    # Memory extraction happens after the response is sent (Phase 2).
    background_tasks.add_task(
        memorize_turn,
        request.app.state.llm,
        request.app.state.embedder,
        session_factory,
        user_message=req.message,
        answer=result["answer"],
        source=str(conversation_id),
    )

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result["answer"],
        route=result.get("route", "general"),
        recalled_memories=result.get("recalled_memories", []),
        sources=[c["source"] for c in result.get("context_chunks", [])],
        confidence=result.get("confidence"),
        unsupported_claims=result.get("unsupported_claims", []),
    )


@app.post("/documents", response_model=IngestResponse)
async def ingest(
    req: IngestRequest, request: Request, user: str = Depends(require_auth)
) -> IngestResponse:
    async with get_session_factory()() as session:
        document_id, chunk_count = await ingest_document(
            session,
            request.app.state.embedder,
            title=req.title,
            text=req.text,
            source=req.source,
        )
    return IngestResponse(document_id=document_id, chunks=chunk_count)


@app.post("/repos", response_model=RepoIngestResponse)
async def ingest_repository(
    req: RepoIngestRequest, request: Request, user: str = Depends(require_auth)
) -> RepoIngestResponse:
    settings = request.app.state.settings
    root = Path(req.path).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {root}")
    async with get_session_factory()() as session:
        stats = await ingest_repo(
            session,
            request.app.state.embedder,
            root,
            max_files=settings.repo_max_files,
            max_file_bytes=settings.repo_max_file_bytes,
        )
    return RepoIngestResponse(
        files_ingested=stats.files_ingested,
        files_skipped=stats.files_skipped,
        chunks=stats.chunks,
    )


@app.post("/vision/analyze", response_model=VisionResponse)
async def vision_analyze(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form("Describe this image in detail."),
    ingest_result: bool = Form(False),
    user: str = Depends(require_auth),
) -> VisionResponse:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="upload must be an image")
    data = await file.read()
    answer = await request.app.state.llm.complete_vision(
        image=data, media_type=file.content_type, question=question
    )

    document_id = None
    if ingest_result and answer:
        async with get_session_factory()() as session:
            document_id, _ = await ingest_document(
                session,
                request.app.state.embedder,
                title=f"vision: {file.filename or 'image'}",
                text=answer,
                source="vision",
            )
    return VisionResponse(answer=answer, ingested_document_id=document_id)


@app.get("/memories/search", response_model=list[MemoryOut])
async def search_memories(
    q: str, request: Request, limit: int = 10, user: str = Depends(require_auth)
) -> list[MemoryOut]:
    async with get_session_factory()() as session:
        memories = await recall(session, request.app.state.embedder, q, limit=limit)
    return [
        MemoryOut(id=m.id, type=m.type.value, content=m.content, access_count=m.access_count)
        for m in memories
    ]
