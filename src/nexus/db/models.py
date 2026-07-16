import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from nexus.config import get_settings

_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner: Mapped[str] = mapped_column(Text, default="anonymous", index=True)
    title: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(Text)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class MemoryType(enum.Enum):
    episodic = "episodic"      # events: "user fixed the auth bug on July 14"
    semantic = "semantic"      # facts: "the backend uses FastAPI"
    procedural = "procedural"  # how-tos: "deploy with `make deploy staging`"


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner: Mapped[str] = mapped_column(Text, default="anonymous", index=True)
    type: Mapped[MemoryType] = mapped_column(Enum(MemoryType, name="memory_type"))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)  # e.g. conversation id it came from
    embedding: Mapped[list[float]] = mapped_column(Vector(_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Access stats power future memory decay/reinforcement without a schema change.
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    access_count: Mapped[int] = mapped_column(Integer, default=0)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner: Mapped[str] = mapped_column(Text, default="anonymous", index=True)
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text, default="text")  # "text" | "code"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(_DIM))

    document: Mapped[Document] = relationship(back_populates="chunks")
