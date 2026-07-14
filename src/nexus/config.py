from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://nexus:nexus@localhost:5432/nexus"

    llm_model: str = "claude-opus-4-8"

    # "hashing" needs no API key (lexical-only vectors, dev/test).
    # "voyage" requires VOYAGE_API_KEY and gives real semantic embeddings.
    embedding_provider: str = "hashing"
    embedding_dim: int = 1024
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-3.5"

    max_context_chunks: int = 6
    max_recalled_memories: int = 5
    max_history_messages: int = 20

    # Phase 2 — self-improving RAG
    rewrite_enabled: bool = True
    rerank_enabled: bool = True
    grading_enabled: bool = True

    # Phase 2 — auth. Unset = auth disabled (single-user dev mode).
    auth_secret: str | None = None
    auth_token_ttl_minutes: int = 60

    # Phase 3 — research agent (server-side web search/fetch)
    research_max_continuations: int = 5

    # Phase 4 — repo ingestion guards
    repo_max_files: int = 2000
    repo_max_file_bytes: int = 200_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
