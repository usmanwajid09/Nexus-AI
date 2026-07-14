import pytest

from nexus.rag.chunker import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_paragraphs_pack_together():
    text = "para one\n\npara two\n\npara three"
    assert chunk_text(text, max_chars=100) == [text]


def test_respects_max_chars():
    text = "\n\n".join(f"paragraph number {i} " + "x" * 200 for i in range(10))
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)


def test_no_content_lost():
    paragraphs = [f"unique-token-{i}" for i in range(50)]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=120)
    joined = "\n".join(chunks)
    assert all(p in joined for p in paragraphs)


def test_long_paragraph_hard_split_has_overlap():
    text = "abcdefghij" * 100  # 1000 chars, no paragraph breaks
    chunks = chunk_text(text, max_chars=300, overlap=100)
    assert len(chunks) > 1
    # consecutive pieces share the overlap region
    assert chunks[0][-100:] == chunks[1][:100]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_small_max_chars_with_default_overlap_is_valid():
    # overlap is clamped internally; a small max_chars must not raise
    chunks = chunk_text("word " * 100, max_chars=50)
    assert chunks and all(len(c) <= 50 for c in chunks)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", overlap=-1)
