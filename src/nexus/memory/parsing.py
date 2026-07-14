"""Validation of LLM memory-extraction output. Pure — no I/O, fully unit-testable."""

from dataclasses import dataclass
from typing import Any

VALID_TYPES = frozenset({"episodic", "semantic", "procedural"})


@dataclass(frozen=True)
class ExtractedMemory:
    type: str
    content: str


def parse_memory_payload(payload: Any) -> list[ExtractedMemory]:
    """Turn the extractor LLM's JSON payload into validated memories.

    Defensive by design: malformed entries are skipped, never raised — a bad
    extraction should degrade to "no memory saved", not fail the chat turn.
    """
    if not isinstance(payload, dict):
        return []
    entries = payload.get("memories")
    if not isinstance(entries, list):
        return []

    memories: list[ExtractedMemory] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mem_type = entry.get("type")
        content = entry.get("content")
        if mem_type not in VALID_TYPES:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        memories.append(ExtractedMemory(type=mem_type, content=content.strip()))
    return memories
