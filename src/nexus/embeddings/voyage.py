import voyageai

from nexus.embeddings.base import EmbeddingProvider


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embeddings (Anthropic's recommended embeddings partner)."""

    def __init__(self, api_key: str | None, model: str = "voyage-3.5", dim: int = 1024) -> None:
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._model = model
        self.dim = dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(
            texts, model=self._model, input_type="document", output_dimension=self.dim
        )
        return result.embeddings

    async def embed_query(self, text: str) -> list[float]:
        result = await self._client.embed(
            [text], model=self._model, input_type="query", output_dimension=self.dim
        )
        return result.embeddings[0]
