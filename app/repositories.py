"""Persistence operations for logical documents and immutable versions."""

from typing import Optional
from uuid import UUID

import numpy as np
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db_models import (
    ChunkRecord,
    ConversationRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentVersionRecord,
    GeneratedAssetRecord,
    MessageRecord,
    MessageSourceRecord,
)
from app.models import Document, DocumentChunk, DocumentStatus


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def find_version_by_hash(self, content_hash: str) -> Optional[DocumentVersionRecord]:
        return self.session.scalar(
            select(DocumentVersionRecord).where(
                DocumentVersionRecord.content_hash == content_hash
            )
        )

    find_by_hash = find_version_by_hash

    def get_version(self, version_id: object) -> Optional[DocumentVersionRecord]:
        return self.session.get(DocumentVersionRecord, UUID(str(version_id)))

    def find_document_by_name(self, name: str) -> Optional[DocumentRecord]:
        return self.session.scalar(
            select(DocumentRecord).where(DocumentRecord.name == name)
        )

    def find_latest_version(self, filename: str) -> Optional[DocumentVersionRecord]:
        return self.session.scalar(
            select(DocumentVersionRecord)
            .where(
                DocumentVersionRecord.filename == filename,
                DocumentVersionRecord.is_current.is_(True),
            )
            .order_by(DocumentVersionRecord.version_number.desc())
        )

    def get_ready(self, version_id: object) -> Optional[DocumentVersionRecord]:
        return self.session.scalar(
            select(DocumentVersionRecord).where(
                DocumentVersionRecord.id == UUID(str(version_id)),
                DocumentVersionRecord.status == DocumentStatus.READY.value,
            )
        )

    def get_current_version(self, document_id: object) -> Optional[DocumentVersionRecord]:
        version = self.get_version(document_id)
        logical_id = version.document_id if version else UUID(str(document_id))
        return self.session.scalar(
            select(DocumentVersionRecord)
            .where(
                DocumentVersionRecord.document_id == logical_id,
                DocumentVersionRecord.is_current.is_(True),
            )
            .order_by(DocumentVersionRecord.version_number.desc())
        )

    def list_versions(self, document_id: object) -> list[DocumentVersionRecord]:
        version = self.get_version(document_id)
        logical_id = version.document_id if version else UUID(str(document_id))
        return list(
            self.session.scalars(
                select(DocumentVersionRecord)
                .where(DocumentVersionRecord.document_id == logical_id)
                .order_by(DocumentVersionRecord.version_number.desc())
            )
        )

    def mark_current_version(self, version_id: object) -> DocumentVersionRecord:
        version = self.get_version(version_id)
        if version is None:
            raise ValueError("Document version was not found")
        self.session.execute(
            update(DocumentVersionRecord)
            .where(DocumentVersionRecord.document_id == version.document_id)
            .values(is_current=False)
        )
        version.is_current = True
        self.session.commit()
        self.session.refresh(version)
        return version

    def get_chunk_texts(self, version_id: object) -> list[str]:
        chunks = self.session.scalars(
            select(ChunkRecord)
            .where(ChunkRecord.document_version_id == UUID(str(version_id)))
            .order_by(ChunkRecord.page_number, ChunkRecord.chunk_index)
        )
        return [f"[Document page {chunk.page_number}]\n{chunk.content}" for chunk in chunks]

    def _build_record(
        self,
        document: Document,
        *,
        logical_id: UUID,
        version_number: int,
    ) -> DocumentVersionRecord:
        if not document.pages or not document.chunks:
            raise ValueError("A document must have pages and chunks before persistence")
        record = DocumentVersionRecord(
            document_id=logical_id,
            version_number=version_number,
            filename=document.filename,
            content_hash=document.document_hash,
            size_bytes=document.size_bytes,
            pdf_data=document.pdf_data,
            page_count=len(document.pages),
            status=document.status.value,
            error=document.error,
            is_current=False,
        )
        self.session.add(record)
        self.session.flush()
        record.pages = [
            DocumentPageRecord(
                document_id=logical_id,
                document_version_id=record.id,
                page_number=page.page_number,
                content=page.text,
            )
            for page in document.pages
        ]
        record.chunks = [
            ChunkRecord(
                id=f"{record.id}-page_{chunk.page_number}-chunk_{chunk.chunk_index}",
                document_id=logical_id,
                document_version_id=record.id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_hash=chunk.content_hash,
            )
            for chunk in document.chunks
        ]
        return record

    def create_document_version(self, document: Document) -> DocumentVersionRecord:
        latest = self.find_latest_version(document.filename)
        if latest is None:
            logical = self.find_document_by_name(document.filename)
            if logical is None:
                logical = DocumentRecord(
                    name=document.filename,
                    filename=document.filename,
                )
                self.session.add(logical)
                self.session.flush()
            logical_id = logical.id
            version_number = 1
        else:
            logical_id = latest.document_id
            version_number = latest.version_number + 1
        record = self._build_record(
            document,
            logical_id=logical_id,
            version_number=version_number,
        )
        self.session.commit()
        self.session.refresh(record)
        return record

    create_processing = create_document_version

    def complete(self, version_id: object, embeddings: np.ndarray) -> DocumentVersionRecord:
        record = self.get_version(version_id)
        if record is None:
            raise ValueError("Document version was not found")
        if len(record.chunks) != len(embeddings):
            raise ValueError("Each chunk must have a corresponding embedding")
        self.session.execute(
            update(DocumentVersionRecord)
            .where(
                DocumentVersionRecord.document_id == record.document_id,
                DocumentVersionRecord.id != record.id,
            )
            .values(is_current=False)
        )
        for chunk, embedding in zip(record.chunks, embeddings):
            chunk.embedding = embedding.tolist()
        record.status = DocumentStatus.READY.value
        record.is_current = True
        record.error = None
        self.session.commit()
        self.session.refresh(record)
        return record

    def fail(self, version_id: object, error: str) -> None:
        record = self.get_version(version_id)
        if record is not None:
            record.status = DocumentStatus.FAILED.value
            record.error = error
            self.session.commit()

    def save(self, document: Document, embeddings: np.ndarray) -> DocumentVersionRecord:
        record = self.create_document_version(document)
        return self.complete(record.id, embeddings)

    def load_chunks(self, version_id: object) -> tuple[list[DocumentChunk], np.ndarray]:
        records = self.session.scalars(
            select(ChunkRecord)
            .where(ChunkRecord.document_version_id == UUID(str(version_id)))
            .order_by(ChunkRecord.page_number, ChunkRecord.chunk_index)
        ).all()
        chunks = [
            DocumentChunk(
                id=record.id,
                document_id=str(version_id),
                page_number=record.page_number,
                chunk_index=record.chunk_index,
                content=record.content,
                content_hash=record.content_hash,
            )
            for record in records
        ]
        return chunks, np.asarray([record.embedding for record in records])

    def delete(self, document_id: object) -> None:
        version = self.get_version(document_id)
        logical = self.session.get(
            DocumentRecord,
            version.document_id if version else UUID(str(document_id)),
        )
        if logical is None:
            raise ValueError("Document was not found")
        self.session.delete(logical)
        self.session.commit()

    def list_documents(self) -> list[DocumentVersionRecord]:
        return list(
            self.session.scalars(
                select(DocumentVersionRecord)
                .where(DocumentVersionRecord.is_current.is_(True))
                .order_by(DocumentVersionRecord.created_at.desc())
            )
        )

    def save_asset(
        self, version_id: object, asset_type: str, payload: dict
    ) -> GeneratedAssetRecord:
        version = self.get_version(version_id)
        if version is None:
            raise ValueError("Document version was not found")
        asset = GeneratedAssetRecord(
            document_id=version.document_id,
            document_version_id=version.id,
            asset_type=asset_type,
            payload=payload,
        )
        self.session.add(asset)
        self.session.commit()
        return asset

    def list_assets(self, document_id: object) -> list[GeneratedAssetRecord]:
        version = self.get_version(document_id)
        logical_id = version.document_id if version else UUID(str(document_id))
        return list(
            self.session.scalars(
                select(GeneratedAssetRecord)
                .where(GeneratedAssetRecord.document_id == logical_id)
                .order_by(GeneratedAssetRecord.created_at.desc())
            )
        )

    def create_conversation(
        self,
        version_id: object,
        conversation_id: Optional[object] = None,
        title: str = "New conversation",
    ) -> ConversationRecord:
        version = self.get_version(version_id)
        if version is None:
            raise ValueError("Document version was not found")
        if conversation_id is not None:
            conversation = self.session.get(
                ConversationRecord, UUID(str(conversation_id))
            )
            if conversation is None or conversation.document_version_id != version.id:
                raise ValueError("Conversation is not pinned to this document version")
            return conversation
        conversation = ConversationRecord(
            document_id=version.document_id,
            document_version_id=version.id,
            title=title,
        )
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def list_conversations(
        self, document_version_id: Optional[object] = None
    ) -> list[ConversationRecord]:
        statement = select(ConversationRecord).order_by(
            ConversationRecord.created_at.desc()
        )
        if document_version_id is not None:
            statement = statement.where(
                ConversationRecord.document_version_id == UUID(str(document_version_id))
            )
        return list(self.session.scalars(statement))

    def delete_conversation(self, conversation_id: object) -> None:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError("Conversation was not found")
        self.session.delete(conversation)
        self.session.commit()

    def get_conversation(self, conversation_id: object) -> Optional[ConversationRecord]:
        return self.session.get(ConversationRecord, UUID(str(conversation_id)))

    def get_conversation_history(
        self, conversation_id: object, *, limit: int = 12
    ) -> str:
        messages = self.session.scalars(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == UUID(str(conversation_id)))
            .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
            .limit(limit)
        ).all()
        return "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in reversed(messages)
            if message.status == "completed"
        )

    def add_message(
        self,
        conversation_id: object,
        role: str,
        content: str,
        sources: Optional[list[object]] = None,
        document_version: int = 1,
        status: str = "pending",
    ) -> MessageRecord:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError("Conversation was not found")
        message = MessageRecord(
            conversation_id=UUID(str(conversation_id)),
            role=role,
            content=content,
            status=status,
            completed_at=datetime.now(timezone.utc) if status == "completed" else None,
        )
        self.session.add(message)
        self.session.flush()
        self.add_message_sources(message, sources or [], document_version)
        self.session.commit()
        self.session.refresh(message)
        return message

    save_message = add_message

    def update_message(
        self,
        message_id: object,
        *,
        content: Optional[str] = None,
        status: Optional[str] = None,
    ) -> MessageRecord:
        message = self.session.get(MessageRecord, UUID(str(message_id)))
        if message is None:
            raise ValueError("Message was not found")
        if content is not None:
            message.content = content
        if status is not None:
            message.status = status
            message.completed_at = (
                datetime.now(timezone.utc) if status == "completed" else None
            )
        self.session.commit()
        self.session.refresh(message)
        return message

    def list_messages(self, conversation_id: object) -> list[MessageRecord]:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError("Conversation was not found")
        return list(
            self.session.scalars(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation.id)
                .order_by(MessageRecord.created_at, MessageRecord.id)
            )
        )

    def add_message_sources(
        self,
        message: MessageRecord,
        sources: list[object],
        document_version: int,
    ) -> None:
        for source in sources:
            version = self.get_version(source.document_id)
            if version is None:
                continue
            message.sources.append(
                MessageSourceRecord(
                    chunk_id=source.chunk_id,
                    document_id=version.document_id,
                    document_version_id=version.id,
                    document_version=document_version,
                    page_number=source.page_number,
                    score=source.score,
                )
            )
