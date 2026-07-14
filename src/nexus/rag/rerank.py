"""Reranking (Phase 2).

A listwise LLM reranker replaces the classic cross-encoder: no torch/native
dependencies (which this project avoids deliberately — locked-down machines
block unsigned native DLLs), and modern LLM rerankers are competitive.
parse_ranking is pure and unit-tested; any failure degrades to original order.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from nexus.llm.base import LLMProvider

logger = logging.getLogger("nexus.rag")

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["ranking"],
    "additionalProperties": False,
}

_RERANK_PROMPT = """\
Rank the passages below by how useful they are for answering the query. \
Return the ranking as a list of passage indices, most useful first. Include \
every index exactly once.

Query: {query}

Passages:
{passages}
"""


def parse_ranking(payload: Any, n_candidates: int) -> list[int]:
    """Validate an LLM ranking: keep in-range unique indices, then append any
    the model forgot in original order, so the result is always a permutation."""
    order: list[int] = []
    if isinstance(payload, dict) and isinstance(payload.get("ranking"), list):
        for entry in payload["ranking"]:
            if isinstance(entry, int) and 0 <= entry < n_candidates and entry not in order:
                order.append(entry)
    for i in range(n_candidates):
        if i not in order:
            order.append(i)
    return order


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, passages: list[str]) -> list[int]:
        """Return a permutation of indices, most relevant first."""


class NoopReranker(Reranker):
    async def rerank(self, query: str, passages: list[str]) -> list[int]:
        return list(range(len(passages)))


class LLMReranker(Reranker):
    def __init__(self, llm: LLMProvider, *, max_passage_chars: int = 800) -> None:
        self._llm = llm
        self._max_passage_chars = max_passage_chars

    async def rerank(self, query: str, passages: list[str]) -> list[int]:
        if len(passages) <= 1:
            return list(range(len(passages)))
        rendered = "\n\n".join(
            f"[{i}] {p[: self._max_passage_chars]}" for i, p in enumerate(passages)
        )
        try:
            payload = await self._llm.complete_json(
                prompt=_RERANK_PROMPT.format(query=query, passages=rendered),
                schema=RERANK_SCHEMA,
                max_tokens=1024,
            )
        except Exception:
            logger.exception("rerank failed; keeping fused order")
            return list(range(len(passages)))
        return parse_ranking(payload, len(passages))
