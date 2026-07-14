from nexus.rag.fusion import rrf_fuse


def test_item_in_both_lists_wins():
    vector = ["a", "b", "c"]
    keyword = ["d", "b", "e"]
    fused = rrf_fuse([vector, keyword])
    assert fused[0] == "b"


def test_single_list_preserves_order():
    assert rrf_fuse([["x", "y", "z"]]) == ["x", "y", "z"]


def test_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_all_items_present_exactly_once():
    fused = rrf_fuse([["a", "b"], ["b", "c"], ["c", "a"]])
    assert sorted(fused) == ["a", "b", "c"]


def test_earlier_rank_beats_later_rank():
    fused = rrf_fuse([["first", "second"], ["first", "second"]])
    assert fused == ["first", "second"]
