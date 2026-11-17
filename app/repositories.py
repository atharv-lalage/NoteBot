"""Persistence operations for documents and their searchable chunks."""

from typing import Optional
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db_models import ChunkRecord, DocumentRecord, DocumentPageRecord, GeneratedAssetRecord
from app.models import Document, DocumentChunk, DocumentPage, DocumentStatus


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def find_by_hash(self, content_hash: str) -> Optional[DocumentRecord]:
        return self.session.scalar(
            select(DocumentRecord).where(DocumentRecord.content_hash == content_hash)
        )

    def get_ready(self, document_id: str) -> Optional[DocumentRecord]:
        parsed_id = UUID(str(document_id))
        return self.session.scalar(
            select(DocumentRecord).where(
                DocumentRecord.id == parsed_id,
                DocumentRecord.status == DocumentStatus.READY.value,
            )
        )

    def get_chunk_texts(self, document_id: str) -> list[str]:
        chunks = self.session.scalars(
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.page_number, ChunkRecord.chunk_index)
        )
        return [f"[Document page {chunk.page_number}]\n{chunk.content}" for chunk in chunks]

    def _build_record(self, document: Document) -> DocumentRecord:
        if not document.pages or not document.chunks:
            raise ValueError("A document must have pages and chunks before persistence")
        record = DocumentRecord(
            filename=document.filename,
            content_hash=document.document_hash,
            size_bytes=document.size_bytes,
            page_count=len(document.pages),
            status=document.status.value,
            error=document.error,
        )
        record.pages = [
            DocumentPageRecord(page_number=page.page_number, content=page.text)
            for page in document.pages
        ]
        record.chunks = [
            ChunkRecord(
                id=chunk.id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_hash=chunk.content_hash,
            )
            for index, chunk in enumerate(document.chunks)
        ]
        return record

    def create_processing(self, document: Document) -> DocumentRecord:
        record = self._build_record(document)
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def complete(self, document_id: object, embeddings: np.ndarray) -> DocumentRecord:
        record = self.session.get(DocumentRecord, document_id)
        if record is None:
            raise ValueError("Document was not found")
        if len(record.chunks) != len(embeddings):
            raise ValueError("Each chunk must have a corresponding embedding")
        for chunk, embedding in zip(record.chunks, embeddings):
            chunk.embedding = embedding.tolist()
        record.status = DocumentStatus.READY.value
        record.error = None
        self.session.commit()
        self.session.refresh(record)
        return record

    def fail(self, document_id: object, error: str) -> None:
        record = self.session.get(DocumentRecord, document_id)
        if record is not None:
            record.status = DocumentStatus.FAILED.value
            record.error = error
            self.session.commit()

    def save(self, document: Document, embeddings: np.ndarray) -> DocumentRecord:
        record = self.create_processing(document)
        return self.complete(record.id, embeddings)

    def load_chunks(self, document_id: object) -> tuple[list[DocumentChunk], np.ndarray]:
        records = self.session.scalars(
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.page_number, ChunkRecord.chunk_index)
        ).all()
        chunks = [
            DocumentChunk(
                id=record.id,
                document_id=str(document_id),
                page_number=record.page_number,
                chunk_index=record.chunk_index,
                content=record.content,
                content_hash=record.content_hash,
            )
            for record in records
        ]
        embeddings = np.asarray([record.embedding for record in records])
        return chunks, embeddings

    def delete(self, document_id: object) -> None:
        record = self.session.get(DocumentRecord, UUID(str(document_id)))
        if record is None:
            raise ValueError("Document was not found")
        self.session.delete(record)
        self.session.commit()

    def list_documents(self) -> list[DocumentRecord]:
        return list(self.session.scalars(select(DocumentRecord).order_by(DocumentRecord.created_at.desc())))

    def save_asset(self, document_id: object, asset_type: str, payload: dict) -> GeneratedAssetRecord:
        asset = GeneratedAssetRecord(
            document_id=UUID(str(document_id)),
            asset_type=asset_type,
            payload=payload,
        )
        self.session.add(asset)
        self.session.commit()
        return asset
