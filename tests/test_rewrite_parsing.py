from nexus.rag.rewrite import parse_queries


def test_valid_queries():
    payload = {"queries": ["fastapi auth flow", "jwt middleware"]}
    assert parse_queries(payload) == ["fastapi auth flow", "jwt middleware"]


def test_dedupes_and_strips():
    payload = {"queries": ["  q1  ", "q1", "q2"]}
    assert parse_queries(payload) == ["q1", "q2"]


def test_caps_at_max():
    payload = {"queries": [f"q{i}" for i in range(10)]}
    assert parse_queries(payload, max_queries=3) == ["q0", "q1", "q2"]


def test_garbage_gives_empty():
    assert parse_queries(None) == []
    assert parse_queries({"queries": "one string"}) == []
    assert parse_queries({"queries": [1, None, ""]}) == []


def test_too_short_and_too_long_queries_dropped():
    payload = {"queries": ["x", "a valid query", "y" * 600]}
    assert parse_queries(payload) == ["a valid query"]


def test_min_len_boundary():
    # min_len defaults to 2 - a two-char query is kept, one-char is dropped
    assert parse_queries({"queries": ["ok", "a"]}) == ["ok"]
