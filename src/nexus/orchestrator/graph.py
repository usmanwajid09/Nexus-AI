"""The Nexus agent graph (Phases 1-4).

    START -> route -+-> research ------------------------------> END
                    |
                    +-> recall -> rewrite -> retrieve -> generate -> grade -> END
                        (general and code routes; code retrieves kind="code")

Memory writing runs as a background task in the API layer (Phase 2), so it no
longer adds latency to the response.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nexus.config import Settings
from nexus.embeddings.base import EmbeddingProvider
from nexus.llm.base import LLMProvider
from nexus.memory.store import recall
from nexus.orchestrator.router import classify_route
from nexus.observability import ANSWER_CONFIDENCE, RETRIEVAL_CHUNKS
from nexus.rag.grading import grade_answer
from nexus.rag.rerank import LLMReranker, NoopReranker
from nexus.rag.retriever import retrieve
from nexus.rag.rewrite import rewrite_query

logger = logging.getLogger("nexus.orchestrator")

SYSTEM_PROMPT = """\
You are Nexus, an AI assistant with long-term memory and access to the user's \
knowledge base.

{memory_block}
{context_block}
Answer the user directly and concisely. When your answer draws on a knowledge \
base excerpt, cite it inline like [1]. If neither your memories nor the \
excerpts cover the question, say so instead of guessing.
{route_addendum}"""

CODE_ADDENDUM = """
The excerpts above are source code from the user's repository. When asked to \
change code, propose the change as a unified diff against the shown code and \
explain it briefly. Never invent code that is not consistent with the excerpts.\
"""

RESEARCH_SYSTEM = """\
You are Nexus's research agent. Use web search and web fetch to answer with \
current information. Cross-check important claims across at least two sources \
when feasible, cite source URLs inline, and clearly say when sources disagree \
or when you could not verify something.\
"""


class NexusState(TypedDict, total=False):
    conversation_id: str
    user_message: str
    history: list[dict[str, str]]  # prior turns, chat format
    route: str  # general | research | code
    search_queries: list[str]
    recalled_memories: list[str]
    context_chunks: list[dict[str, str]]  # {"content", "source"}
    answer: str
    confidence: float | None
    unsupported_claims: list[str]


def build_graph(
    *,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
):
    reranker = LLMReranker(llm) if settings.rerank_enabled else NoopReranker()

    async def route_node(state: NexusState) -> NexusState:
        return {"route": await classify_route(llm, state["user_message"])}

    async def recall_node(state: NexusState) -> NexusState:
        async with session_factory() as session:
            memories = await recall(
                session, embedder, state["user_message"], limit=settings.max_recalled_memories
            )
        return {"recalled_memories": [m.content for m in memories]}

    async def rewrite_node(state: NexusState) -> NexusState:
        if not settings.rewrite_enabled:
            return {"search_queries": [state["user_message"]]}
        queries = await rewrite_query(
            llm, question=state["user_message"], history=state.get("history", [])
        )
        return {"search_queries": queries}

    async def retrieve_node(state: NexusState) -> NexusState:
        kind = "code" if state.get("route") == "code" else None
        async with session_factory() as session:
            chunks = await retrieve(
                session,
                embedder,
                state.get("search_queries") or [state["user_message"]],
                limit=settings.max_context_chunks,
                kind=kind,
                reranker=reranker,
                rerank_query=state["user_message"],
            )
        RETRIEVAL_CHUNKS.observe(len(chunks))
        return {
            "context_chunks": [
                {"content": c.content, "source": c.document_title} for c in chunks
            ]
        }

    async def generate_node(state: NexusState) -> NexusState:
        memories = state.get("recalled_memories", [])
        chunks = state.get("context_chunks", [])

        memory_block = ""
        if memories:
            lines = "\n".join(f"- {m}" for m in memories)
            memory_block = f"Things you remember about this user and their work:\n{lines}\n"

        context_block = ""
        if chunks:
            lines = "\n\n".join(
                f"[{i + 1}] (from \"{c['source']}\")\n{c['content']}"
                for i, c in enumerate(chunks)
            )
            context_block = f"Knowledge base excerpts relevant to the question:\n{lines}\n"

        route_addendum = CODE_ADDENDUM if state.get("route") == "code" else ""
        messages = [*state.get("history", []), {"role": "user", "content": state["user_message"]}]
        answer = await llm.complete(
            system=SYSTEM_PROMPT.format(
                memory_block=memory_block,
                context_block=context_block,
                route_addendum=route_addendum,
            ),
            messages=messages,
        )
        return {"answer": answer}

    async def research_node(state: NexusState) -> NexusState:
        messages = [*state.get("history", []), {"role": "user", "content": state["user_message"]}]
        try:
            answer = await llm.research(system=RESEARCH_SYSTEM, messages=messages)
        except NotImplementedError:
            logger.warning("provider lacks web research; answering without it")
            answer = await llm.complete(system=RESEARCH_SYSTEM, messages=messages)
        return {"answer": answer, "recalled_memories": [], "context_chunks": []}

    async def grade_node(state: NexusState) -> NexusState:
        chunks = state.get("context_chunks", [])
        if not settings.grading_enabled or not chunks:
            return {"confidence": None, "unsupported_claims": []}
        grade = await grade_answer(
            llm,
            question=state["user_message"],
            answer=state.get("answer", ""),
            context=[c["content"] for c in chunks],
        )
        if grade is None:
            return {"confidence": None, "unsupported_claims": []}
        ANSWER_CONFIDENCE.observe(grade.score)
        return {"confidence": grade.score, "unsupported_claims": grade.unsupported_claims}

    builder = StateGraph(NexusState)
    builder.add_node("route", route_node)
    builder.add_node("research", research_node)
    builder.add_node("recall", recall_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("grade", grade_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: state.get("route", "general"),
        {"research": "research", "general": "recall", "code": "recall"},
    )
    builder.add_edge("research", END)
    builder.add_edge("recall", "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "grade")
    builder.add_edge("grade", END)
    return builder.compile()
