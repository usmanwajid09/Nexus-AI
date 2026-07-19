import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, select, text

from nexus.api.auth import require_auth
from nexus.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    IngestRequest,
    IngestResponse,
    MemoryOut,
    MessageOut,
    RepoIngestRequest,
    RepoIngestResponse,
    VisionResponse,
)
from nexus.api.streaming import sse_format
from nexus.code.ingest import ingest_repo
from nexus.config import get_settings
from nexus.db.models import Chunk, Conversation, Document, Memory, Message
from nexus.db.session import get_session_factory, init_db
from nexus.embeddings import get_embedder
from nexus.llm import get_llm
from nexus.memory.store import recall
from nexus.memory.writer import memorize_turn
from nexus.observability import REQUEST_LATENCY, metrics_app
from nexus.orchestrator.graph import build_graph
from nexus.rag.ingest import ingest_document

logger = logging.getLogger("nexus")

# Fire-and-forget tasks (post-stream memory writes) need a strong reference
# until they finish, or the event loop may garbage-collect them mid-flight.
_pending_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


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


app = FastAPI(title="Nexus AI", version="0.9.0", lifespan=lifespan)
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


_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def web_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness + database reachability check."""
    db_ok = False
    try:
        async with get_session_factory()() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        logger.exception("database health check failed")
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}


@app.get("/version")
async def version(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    return {
        "version": app.version,
        "model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_dim": settings.embedding_dim,
        "features": {
            "rewrite": settings.rewrite_enabled,
            "rerank": settings.rerank_enabled,
            "grading": settings.grading_enabled,
            "auth": bool(settings.auth_secret),
        },
    }


async def _start_turn(
    req: ChatRequest, settings, owner: str
) -> tuple[uuid.UUID, list[dict[str, str]]]:
    """Resolve/create the conversation, load history, persist the user message."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        if req.conversation_id is not None:
            conversation = await session.get(Conversation, req.conversation_id)
            # 404 (not 403) on foreign conversations: don't leak their existence.
            if conversation is None or conversation.owner != owner:
                raise HTTPException(status_code=404, detail="conversation not found")
        else:
            conversation = Conversation(title=req.message[:80], owner=owner)
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
        return conversation.id, history


async def _finish_turn(
    app_state, *, conversation_id: uuid.UUID, user_message: str, answer: str, owner: str
) -> None:
    """Persist the assistant message and kick off background memory extraction."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        session.add(Message(conversation_id=conversation_id, role="assistant", content=answer))
        await session.commit()
    _spawn(
        memorize_turn(
            app_state.llm,
            app_state.embedder,
            session_factory,
            user_message=user_message,
            answer=answer,
            source=str(conversation_id),
            owner=owner,
        )
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    user: str = Depends(require_auth),
) -> ChatResponse:
    conversation_id, history = await _start_turn(req, request.app.state.settings, user)

    result = await request.app.state.graph.ainvoke(
        {
            "conversation_id": str(conversation_id),
            "owner": user,
            "user_message": req.message,
            "history": history,
        }
    )

    await _finish_turn(
        request.app.state,
        conversation_id=conversation_id,
        user_message=req.message,
        answer=result["answer"],
        owner=user,
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


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    user: str = Depends(require_auth),
) -> StreamingResponse:
    """SSE variant of /chat.

    Events: meta (conversation id) -> route -> sources -> delta* -> grade -> done.
    The research route emits its whole answer as one delta.
    """
    conversation_id, history = await _start_turn(req, request.app.state.settings, user)
    graph = request.app.state.graph
    app_state = request.app.state
    owner = user

    async def event_stream():
        final: dict = {}
        parts: list[str] = []
        yield sse_format("meta", {"conversation_id": str(conversation_id)})
        try:
            async for mode, chunk in graph.astream(
                {
                    "conversation_id": str(conversation_id),
                    "owner": owner,
                    "user_message": req.message,
                    "history": history,
                },
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    if isinstance(chunk, dict) and chunk.get("type") == "delta":
                        parts.append(chunk["text"])
                        yield sse_format("delta", {"text": chunk["text"]})
                    continue
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    final.update(update)
                    if node == "route":
                        yield sse_format("route", {"route": update.get("route", "general")})
                    elif node == "retrieve":
                        sources = [c["source"] for c in update.get("context_chunks", [])]
                        yield sse_format("sources", {"sources": sources})
                    elif node == "research" and update.get("answer"):
                        parts.append(update["answer"])
                        yield sse_format("delta", {"text": update["answer"]})
                    elif node == "grade":
                        yield sse_format(
                            "grade",
                            {
                                "confidence": update.get("confidence"),
                                "unsupported_claims": update.get("unsupported_claims", []),
                            },
                        )
        except Exception:
            logger.exception("chat stream failed (conversation=%s)", conversation_id)
            yield sse_format("error", {"detail": "internal error"})
            return

        answer = final.get("answer") or "".join(parts)
        await _finish_turn(
            app_state,
            conversation_id=conversation_id,
            user_message=req.message,
            answer=answer,
            owner=owner,
        )
        yield sse_format(
            "done",
            {
                "conversation_id": str(conversation_id),
                "answer": answer,
                "route": final.get("route", "general"),
                "recalled_memories": final.get("recalled_memories", []),
                "sources": [c["source"] for c in final.get("context_chunks", [])],
                "confidence": final.get("confidence"),
                "unsupported_claims": final.get("unsupported_claims", []),
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
            owner=user,
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
            owner=user,
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
                owner=user,
            )
    return VisionResponse(answer=answer, ingested_document_id=document_id)


@app.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    limit: int = 50, offset: int = 0, user: str = Depends(require_auth)
) -> list[ConversationSummary]:
    async with get_session_factory()() as session:
        stmt = (
            select(Conversation, func.count(Message.id).label("message_count"))
            .outerjoin(Message, Message.conversation_id == Conversation.id)
            .where(Conversation.owner == user)
            .group_by(Conversation.id)
            .order_by(Conversation.created_at.desc())
            .limit(min(limit, 200))
            .offset(offset)
        )
        rows = (await session.execute(stmt)).all()
    return [
        ConversationSummary(
            id=conv.id, title=conv.title, created_at=conv.created_at, message_count=count
        )
        for conv, count in rows
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID, user: str = Depends(require_auth)
) -> ConversationDetail:
    async with get_session_factory()() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or conversation.owner != user:
            raise HTTPException(status_code=404, detail="conversation not found")
        messages = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            )
        ).all()
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[
            MessageOut(role=m.role, content=m.content, created_at=m.created_at) for m in messages
        ],
    )


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: uuid.UUID, user: str = Depends(require_auth)
) -> None:
    async with get_session_factory()() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or conversation.owner != user:
            raise HTTPException(status_code=404, detail="conversation not found")
        await session.delete(conversation)  # messages cascade via FK
        await session.commit()


@app.get("/stats")
async def stats(user: str = Depends(require_auth)) -> dict[str, int]:
    """Per-owner counts across the corpus. Cheap; safe to poll from a UI."""
    async with get_session_factory()() as session:
        counts = {
            "conversations": await session.scalar(
                select(func.count()).select_from(Conversation).where(Conversation.owner == user)
            ),
            "documents": await session.scalar(
                select(func.count()).select_from(Document).where(Document.owner == user)
            ),
            "chunks": await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.owner == user)
            ),
            "memories": await session.scalar(
                select(func.count()).select_from(Memory).where(Memory.owner == user)
            ),
        }
    return {k: v or 0 for k, v in counts.items()}


@app.get("/memories/search", response_model=list[MemoryOut])
async def search_memories(
    q: str, request: Request, limit: int = 10, user: str = Depends(require_auth)
) -> list[MemoryOut]:
    async with get_session_factory()() as session:
        memories = await recall(session, request.app.state.embedder, q, owner=user, limit=limit)
    return [
        MemoryOut(id=m.id, type=m.type.value, content=m.content, access_count=m.access_count)
        for m in memories
    ]
