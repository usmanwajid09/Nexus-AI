"""Route classification (Phase 3): decide which specialist handles a message.
parse_route is pure and unit-tested; failures degrade to the general pipeline."""

import logging
from typing import Any

from nexus.llm.base import LLMProvider

logger = logging.getLogger("nexus.orchestrator")

ROUTES = ("general", "research", "code")

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": list(ROUTES)},
    },
    "required": ["route"],
    "additionalProperties": False,
}

_ROUTE_PROMPT = """\
Classify which specialist should handle this user message.

- "research": needs current information from the live web (news, prices, \
recent releases, anything after a knowledge cutoff, or the user explicitly \
asks to search/look up online).
- "code": about source code that has been ingested into the knowledge base — \
how a codebase works, where something is implemented, reviewing or changing \
code.
- "general": everything else (questions answerable from stored memories and \
ingested documents, or plain conversation).

User message:
{question}
"""


def parse_route(payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("route") in ROUTES:
        return payload["route"]
    return "general"


async def classify_route(llm: LLMProvider, question: str) -> str:
    try:
        payload = await llm.complete_json(
            prompt=_ROUTE_PROMPT.format(question=question),
            schema=ROUTE_SCHEMA,
            max_tokens=512,
        )
    except Exception:
        logger.exception("route classification failed; using general")
        return "general"
    return parse_route(payload)
