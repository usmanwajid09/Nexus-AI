"""Alembic environment wired to the Nexus models.

Workflow (run against a live database):
    alembic revision --autogenerate -m "initial schema"
    alembic upgrade head

init_db() (create_all) remains the dev-mode path; once you adopt migrations,
stop relying on create_all for schema changes.
"""

from alembic import context
from sqlalchemy import create_engine, text

from nexus.config import get_settings
from nexus.db.models import Base

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # psycopg3 URLs work synchronously too, so migrations use a plain engine.
    engine = create_engine(get_settings().database_url)
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
