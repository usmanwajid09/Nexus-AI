"""Retrieval evaluation metrics. Pure functions — no I/O, fully unit-tested."""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def hit_at_k(ranked: Sequence[T], relevant: set[T], k: int) -> bool:
    """True if any relevant item appears in the top k results."""
    return any(item in relevant for item in ranked[:k])


def reciprocal_rank(ranked: Sequence[T], relevant: set[T]) -> float:
    """1/rank of the first relevant item, 0.0 if none retrieved."""
    for index, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (index + 1)
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
