from nexus.rag.grading import parse_grade


def test_valid_grade():
    grade = parse_grade({"grounded_score": 85, "unsupported_claims": ["claim x"]})
    assert grade is not None
    assert grade.score == 0.85
    assert grade.unsupported_claims == ["claim x"]


def test_score_clamped():
    assert parse_grade({"grounded_score": 150, "unsupported_claims": []}).score == 1.0
    assert parse_grade({"grounded_score": -10, "unsupported_claims": []}).score == 0.0


def test_garbage_returns_none():
    assert parse_grade(None) is None
    assert parse_grade({}) is None
    assert parse_grade({"grounded_score": "high"}) is None


def test_bad_claims_filtered():
    grade = parse_grade({"grounded_score": 50, "unsupported_claims": [42, "", "  ", "real claim"]})
    assert grade.unsupported_claims == ["real claim"]


def test_non_list_claims_tolerated():
    grade = parse_grade({"grounded_score": 50, "unsupported_claims": "not a list"})
    assert grade is not None
    assert grade.unsupported_claims == []
