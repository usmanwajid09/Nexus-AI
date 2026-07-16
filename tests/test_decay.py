import pytest

from nexus.memory.decay import decay_weight, memory_score


def test_fresh_memory_has_full_weight():
    assert decay_weight(0, 0) == 1.0
    assert decay_weight(-5, 0) == 1.0


def test_half_life():
    assert decay_weight(30, 0, half_life_days=30) == pytest.approx(0.5)
    assert decay_weight(60, 0, half_life_days=30) == pytest.approx(0.25)


def test_weight_decreases_with_age():
    weights = [decay_weight(d, 0) for d in (1, 10, 30, 90, 365)]
    assert weights == sorted(weights, reverse=True)
    assert all(0 < w <= 1 for w in weights)


def test_accesses_slow_the_decay():
    old_unused = decay_weight(60, 0)
    old_reinforced = decay_weight(60, 8)
    assert old_reinforced > old_unused


def test_negative_accesses_treated_as_zero():
    assert decay_weight(30, -3) == decay_weight(30, 0)


def test_invalid_half_life_raises():
    with pytest.raises(ValueError):
        decay_weight(10, 0, half_life_days=0)


def test_memory_score_combines_similarity_and_decay():
    fresh = memory_score(0.9, 0, 0)
    aged = memory_score(0.9, 60, 0, half_life_days=30)
    assert fresh == pytest.approx(0.9)
    assert aged == pytest.approx(0.9 * 0.25)


def test_memory_score_clamps_negative_similarity():
    assert memory_score(-0.4, 10, 0) == 0.0


def test_reinforced_old_memory_can_beat_newer_unused_one():
    old_but_used = memory_score(0.8, 90, 10)
    newer_never_used = memory_score(0.8, 60, 0)
    assert old_but_used > newer_never_used
