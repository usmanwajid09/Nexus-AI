from nexus.evals.metrics import hit_at_k, mean, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) is True
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) is False
    assert hit_at_k([], {"c"}, 5) is False


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "target", "y"], {"target"}) == 0.5
    assert reciprocal_rank(["target"], {"target"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"target"}) == 0.0


def test_mean():
    assert mean([1.0, 0.0]) == 0.5
    assert mean([]) == 0.0
