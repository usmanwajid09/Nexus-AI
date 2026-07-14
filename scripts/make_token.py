"""Mint a JWT for the Nexus API.

Usage:
    python scripts/make_token.py [subject]

Reads AUTH_SECRET from the environment / .env. Pass the token as
`Authorization: Bearer <token>`.
"""

import sys

from nexus.api.auth import create_token
from nexus.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.auth_secret:
        sys.exit("AUTH_SECRET is not set - auth is disabled, no token needed.")
    subject = sys.argv[1] if len(sys.argv) > 1 else "dev-user"
    token = create_token(
        subject, settings.auth_secret, ttl_minutes=settings.auth_token_ttl_minutes
    )
    print(token)


if __name__ == "__main__":
    main()
