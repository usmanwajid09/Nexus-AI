"""JWT bearer auth (HS256).

When AUTH_SECRET is unset the API runs in single-user dev mode and every
request is treated as "anonymous". Setting AUTH_SECRET turns auth on for all
data endpoints; mint tokens with scripts/make_token.py.
"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Header, HTTPException

from nexus.config import get_settings


class TokenError(Exception):
    pass


def create_token(subject: str, secret: str, *, ttl_minutes: int = 60) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": subject, "iat": now, "exp": now + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str, secret: str) -> str:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise TokenError("token has no subject")
    return subject


async def require_auth(authorization: str | None = Header(default=None)) -> str:
    secret = get_settings().auth_secret
    if not secret:
        return "anonymous"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return verify_token(authorization.removeprefix("Bearer "), secret)
    except TokenError:
        raise HTTPException(
            status_code=401,
            detail="invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
