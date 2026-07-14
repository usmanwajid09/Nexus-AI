import hashlib
import math
import re

from nexus.embeddings.base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(EmbeddingProvider):
    """Deterministic bag-of-words vectors via feature hashing.

    Lexical-only (no semantics), but zero-dependency and stable across runs,
    which makes local development and tests possible without an embeddings API
    key. Swap to Voyage in production via EMBEDDING_PROVIDER=voyage.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
            sign = 1.0 if (digest >> 127) & 1 == 0 else -1.0
            vec[digest % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
