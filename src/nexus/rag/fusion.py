"""Reciprocal Rank Fusion. Pure function — no I/O, fully unit-testable."""

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def rrf_fuse(rankings: Sequence[Sequence[T]], *, k: int = 60) -> list[T]:
    """Merge several ranked lists into one, best-first.

    Standard RRF: each item scores sum(1 / (k + rank_i)) over the lists it
    appears in. Items found by multiple retrievers (e.g. both vector and
    keyword search) rise to the top without any score normalization between
    incomparable scoring scales.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)
