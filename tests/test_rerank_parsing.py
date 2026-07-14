from nexus.rag.rerank import parse_ranking


def test_valid_permutation():
    assert parse_ranking({"ranking": [2, 0, 1]}, 3) == [2, 0, 1]


def test_missing_indices_appended_in_original_order():
    assert parse_ranking({"ranking": [3]}, 5) == [3, 0, 1, 2, 4]


def test_out_of_range_and_duplicates_dropped():
    assert parse_ranking({"ranking": [1, 1, 9, -2, 0]}, 3) == [1, 0, 2]


def test_garbage_payload_gives_identity():
    assert parse_ranking(None, 4) == [0, 1, 2, 3]
    assert parse_ranking({"ranking": "nope"}, 2) == [0, 1]
    assert parse_ranking({}, 0) == []


def test_result_is_always_a_permutation():
    result = parse_ranking({"ranking": [5, 2, 2, 100]}, 6)
    assert sorted(result) == list(range(6))
