import json

from nexus.api.streaming import sse_format


def test_frame_shape():
    frame = sse_format("delta", {"text": "hello"})
    assert frame == 'event: delta\ndata: {"text":"hello"}\n\n'


def test_data_is_single_line_even_with_newlines_in_payload():
    frame = sse_format("delta", {"text": "line1\nline2"})
    body = frame.split("data: ", 1)[1].rstrip("\n")
    assert "\n" not in body
    assert json.loads(body) == {"text": "line1\nline2"}


def test_unicode_preserved():
    frame = sse_format("delta", {"text": "héllo — ünïcode"})
    body = frame.split("data: ", 1)[1].rstrip("\n")
    assert json.loads(body)["text"] == "héllo — ünïcode"
