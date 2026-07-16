"""Memory decay and reinforcement (Phase 6). Pure functions — unit-tested.

Modeled on the Ebbinghaus forgetting curve: a memory's weight halves every
`half_life_days`, but each recall stretches the effective half-life
(reinforcement) — frequently used memories fade slower, like human memory.
"""


def decay_weight(
    age_days: float,
    accesses: int,
    *,
    half_life_days: float = 30.0,
    reinforcement: float = 0.25,
) -> float:
    """Return a weight in (0, 1]: 1.0 when fresh, 0.5 after one half-life.

    Each past access multiplies the effective half-life by (1 + reinforcement),
    additively: effective = half_life * (1 + reinforcement * accesses).
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if age_days <= 0:
        return 1.0
    effective_half_life = half_life_days * (1.0 + reinforcement * max(accesses, 0))
    return 0.5 ** (age_days / effective_half_life)


def memory_score(
    similarity: float,
    age_days: float,
    accesses: int,
    *,
    half_life_days: float = 30.0,
    reinforcement: float = 0.25,
) -> float:
    """Combined recall score: semantic similarity discounted by forgetting."""
    return max(similarity, 0.0) * decay_weight(
        age_days, accesses, half_life_days=half_life_days, reinforcement=reinforcement
    )
