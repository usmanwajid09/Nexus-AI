from nexus.config import Settings
from nexus.embeddings.base import EmbeddingProvider


def get_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "voyage":
        from nexus.embeddings.voyage import VoyageEmbeddings

        return VoyageEmbeddings(
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
            dim=settings.embedding_dim,
        )
    from nexus.embeddings.hashing import HashingEmbeddings

    return HashingEmbeddings(dim=settings.embedding_dim)
