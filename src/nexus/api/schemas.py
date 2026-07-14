import uuid

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    answer: str
    route: str
    recalled_memories: list[str]
    sources: list[str]
    confidence: float | None = None
    unsupported_claims: list[str] = []


class IngestRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str | None = None


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    chunks: int


class RepoIngestRequest(BaseModel):
    path: str = Field(min_length=1, description="Local path to a repository checkout")


class RepoIngestResponse(BaseModel):
    files_ingested: int
    files_skipped: int
    chunks: int


class VisionResponse(BaseModel):
    answer: str
    ingested_document_id: uuid.UUID | None = None


class MemoryOut(BaseModel):
    id: uuid.UUID
    type: str
    content: str
    access_count: int
