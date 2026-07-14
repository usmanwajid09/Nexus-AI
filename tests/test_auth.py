import pytest

from nexus.api.auth import TokenError, create_token, verify_token

SECRET = "test-secret-0123456789abcdef0123456789abcdef"


def test_token_round_trip():
    token = create_token("alice", SECRET, ttl_minutes=5)
    assert verify_token(token, SECRET) == "alice"


def test_wrong_secret_rejected():
    token = create_token("alice", SECRET)
    with pytest.raises(TokenError):
        verify_token(token, "other-secret-0123456789abcdef0123456789")


def test_expired_token_rejected():
    token = create_token("alice", SECRET, ttl_minutes=-1)
    with pytest.raises(TokenError):
        verify_token(token, SECRET)


def test_garbage_token_rejected():
    with pytest.raises(TokenError):
        verify_token("not.a.token", SECRET)
