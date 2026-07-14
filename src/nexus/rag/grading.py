"""Answer groundedness grading (Phase 2): an LLM judge scores how well the
answer is supported by the retrieved context, and lists unsupported claims.
parse_grade is pure and unit-tested; failures degrade to "ungraded" (None)."""

import logging
from dataclasses import dataclass, field
from typing import Any

from nexus.llm.base import LLMProvider

logger = logging.getLogger("nexus.rag")

GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded_score": {"type": "integer"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded_score", "unsupported_claims"],
    "additionalProperties": False,
}

_GRADING_PROMPT = """\
You are grading an AI assistant's answer for groundedness.

Question:
{question}

Context the assistant was given:
{context}

Answer being graded:
{answer}

Score grounded_score from 0 to 100: 100 means every factual claim in the \
answer is directly supported by the context (or is trivially common \
knowledge); 0 means the answer is unsupported. List each factual claim that \
the context does not support in unsupported_claims (empty list if none).
"""


@dataclass
class Grade:
    score: float  # 0.0 - 1.0
    unsupported_claims: list[str] = field(default_factory=list)


def parse_grade(payload: Any) -> Grade | None:
    if not isinstance(payload, dict):
        return None
    raw_score = payload.get("grounded_score")
    if not isinstance(raw_score, int):
        return None
    score = min(max(raw_score, 0), 100) / 100.0
    claims_raw = payload.get("unsupported_claims")
    claims = [c.strip() for c in claims_raw if isinstance(c, str) and c.strip()] \
        if isinstance(claims_raw, list) else []
    return Grade(score=score, unsupported_claims=claims)


async def grade_answer(
    llm: LLMProvider, *, question: str, answer: str, context: list[str]
) -> Grade | None:
    try:
        payload = await llm.complete_json(
            prompt=_GRADING_PROMPT.format(
                question=question,
                context="\n\n---\n\n".join(context) or "(no context)",
                answer=answer,
            ),
            schema=GRADING_SCHEMA,
            max_tokens=2048,
        )
    except Exception:
        logger.exception("answer grading failed")
        return None
    return parse_grade(payload)
