from nexus.orchestrator.router import parse_route


def test_valid_routes():
    assert parse_route({"route": "research"}) == "research"
    assert parse_route({"route": "code"}) == "code"
    assert parse_route({"route": "general"}) == "general"


def test_garbage_defaults_to_general():
    assert parse_route(None) == "general"
    assert parse_route({}) == "general"
    assert parse_route({"route": "banana"}) == "general"
    assert parse_route({"route": 3}) == "general"
