"""SQLAlchemy persistence models for NoteBot."""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.database import Base


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[object] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    documents: Mapped[List["DocumentRecord"]] = relationship(back_populates="user")
    conversations: Mapped[List["ConversationRecord"]] = relationship(back_populates="user")


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[object] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[object]] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped[Optional[UserRecord]] = relationship(back_populates="documents")
    pages: Mapped[List["DocumentPageRecord"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["ChunkRecord"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    conversations: Mapped[List["ConversationRecord"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    generated_assets: Mapped[List["GeneratedAssetRecord"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentPageRecord(Base):
    __tablename__ = "document_pages"

    id: Mapped[object] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[object] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    document: Mapped[DocumentRecord] = relationship(back_populates="pages")


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    document_id: Mapped[object] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.embedding_dimensions)
    )
    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")
    message_sources: Mapped[List["MessageSourceRecord"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[object] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[object]] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    document_id: Mapped[object] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped[Optional[UserRecord]] = relationship(back_populates="conversations")
    document: Mapped[DocumentRecord] = relationship(back_populates="conversations")
    messages: Mapped[List["MessageRecord"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[object] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[object] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    conversation: Mapped[ConversationRecord] = relationship(back_populates="messages")
    sources: Mapped[List["MessageSourceRecord"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class MessageSourceRecord(Base):
    __tablename__ = "message_sources"

    id: Mapped[object] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id: Mapped[object] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    score: Mapped[float] = mapped_column(nullable=False)
    message: Mapped[MessageRecord] = relationship(back_populates="sources")
    chunk: Mapped[ChunkRecord] = relationship(back_populates="message_sources")


class GeneratedAssetRecord(Base):
    __tablename__ = "generated_assets"

    id: Mapped[object] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[object] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    asset_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document: Mapped[DocumentRecord] = relationship(back_populates="generated_assets")
