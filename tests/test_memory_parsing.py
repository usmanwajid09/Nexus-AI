from nexus.memory.parsing import ExtractedMemory, parse_memory_payload


def test_valid_payload():
    payload = {
        "memories": [
            {"type": "semantic", "content": "Backend uses FastAPI."},
            {"type": "procedural", "content": "Deploy with make deploy."},
        ]
    }
    assert parse_memory_payload(payload) == [
        ExtractedMemory(type="semantic", content="Backend uses FastAPI."),
        ExtractedMemory(type="procedural", content="Deploy with make deploy."),
    ]


def test_invalid_type_skipped():
    payload = {"memories": [{"type": "banana", "content": "nope"}]}
    assert parse_memory_payload(payload) == []


def test_blank_content_skipped():
    payload = {"memories": [{"type": "semantic", "content": "   "}]}
    assert parse_memory_payload(payload) == []


def test_non_dict_entries_skipped():
    payload = {"memories": ["just a string", 42, None, {"type": "episodic", "content": "ok"}]}
    assert parse_memory_payload(payload) == [ExtractedMemory(type="episodic", content="ok")]


def test_garbage_payloads():
    assert parse_memory_payload(None) == []
    assert parse_memory_payload("nope") == []
    assert parse_memory_payload({}) == []
    assert parse_memory_payload({"memories": "not a list"}) == []


def test_content_is_stripped():
    payload = {"memories": [{"type": "semantic", "content": "  padded  "}]}
    assert parse_memory_payload(payload)[0].content == "padded"
