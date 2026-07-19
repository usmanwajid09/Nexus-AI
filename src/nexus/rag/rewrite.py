"""Query rewriting (Phase 2): turn a conversational question into 1-3 search
queries. parse_queries is pure and unit-tested; the LLM call degrades to the
original question on any failure."""

import logging
from typing import Any

from nexus.llm.base import LLMProvider

logger = logging.getLogger("nexus.rag")

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
    "additionalProperties": False,
}

_REWRITE_PROMPT = """\
Rewrite the user's question into search queries for a knowledge base that uses \
both semantic and keyword search.

Rules:
- 1 to 3 queries; each self-contained (resolve pronouns like "it" or "that" \
using the conversation context below).
- Prefer specific nouns and identifiers over conversational phrasing.
- If the question is already a good query, return it as the single query.

Recent conversation:
{history}

User question:
{question}
"""


def parse_queries(
    payload: Any, *, max_queries: int = 3, min_len: int = 2, max_len: int = 500
) -> list[str]:
    """Extract clean, bounded search queries from a rewriter payload.

    Guards against pathological outputs: single-character queries, novel-length
    prose, non-string entries, duplicates. Any of those would waste retrieval
    work or embed poorly.
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("queries")
    if not isinstance(raw, list):
        return []
    queries: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        cleaned = entry.strip()
        if len(cleaned) < min_len or len(cleaned) > max_len:
            continue
        if cleaned in queries:
            continue
        queries.append(cleaned)
        if len(queries) >= max_queries:
            break
    return queries


async def rewrite_query(
    llm: LLMProvider, *, question: str, history: list[dict[str, str]]
) -> list[str]:
    history_text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-6:]) or "(none)"
    try:
        payload = await llm.complete_json(
            prompt=_REWRITE_PROMPT.format(history=history_text, question=question),
            schema=REWRITE_SCHEMA,
            max_tokens=1024,
        )
    except Exception:
        logger.exception("query rewrite failed; using original question")
        return [question]
    return parse_queries(payload) or [question]
