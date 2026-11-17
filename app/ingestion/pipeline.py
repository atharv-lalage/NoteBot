"""PDF-to-page-aware document ingestion pipeline."""

import hashlib
from typing import BinaryIO, Optional

from app.ingestion.pdf_loader import extract_pages, read_file_bytes
from app.ingestion.chunker import ChunkingConfig, chunk_pages
from app.models import Document, DocumentStatus


class DuplicateDocumentError(ValueError):
    """Raised when the same document has already been processed."""


def calculate_document_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DocumentRegistry:
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def get(self, document_hash: str) -> Optional[Document]:
        return self._documents.get(document_hash)

    def add(self, document: Document) -> None:
        self._documents[document.document_hash] = document

    def clear(self) -> None:
        self._documents.clear()


def process_document(
    file: BinaryIO,
    registry: Optional[DocumentRegistry] = None,
    chunking_config: ChunkingConfig = ChunkingConfig(),
) -> Document:
    data = read_file_bytes(file)
    document_hash = calculate_document_hash(data)
    existing = registry.get(document_hash) if registry else None
    if existing and existing.status in (DocumentStatus.PROCESSING, DocumentStatus.READY):
        raise DuplicateDocumentError(
            f"This document was already uploaded as '{existing.filename}'."
        )
    document = Document(
        document_hash=document_hash,
        filename=getattr(file, "name", "uploaded.pdf"),
        size_bytes=len(data),
        status=DocumentStatus.PROCESSING,
    )
    if registry:
        registry.add(document)
    try:
        document.pages = extract_pages(data)
        document.chunks = chunk_pages(document.pages, document.document_hash, chunking_config)
        document.status = DocumentStatus.READY
        return document
    except Exception as error:
        document.status = DocumentStatus.FAILED
        document.error = str(error)
        if registry:
            registry.add(document)
        raise


def process_pdf(file: BinaryIO) -> list[str]:
    """Backward-compatible helper returning page-aware chunks as plain strings."""
    return [chunk.content for chunk in process_document(file).chunks]
