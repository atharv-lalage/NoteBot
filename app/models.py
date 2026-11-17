"""Domain models for indexed notes."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import UUID


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    page_number: int
    chunk_index: int
    content: str
    content_hash: str

    @property
    def text(self) -> str:
        """Compatibility alias for retrieval and rendering code."""
        return self.content


@dataclass
class Document:
    document_hash: str
    filename: str
    size_bytes: int
    pdf_data: Optional[bytes] = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    pages: Optional[list[DocumentPage]] = None
    chunks: Optional[list[DocumentChunk]] = None
    error: Optional[str] = None
    version_number: int = 1
    version_group_id: Optional[UUID] = None
    is_current: bool = True

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages or [])


@dataclass
class NoteIndex:
    chunks: list[DocumentChunk]
    embeddings: object
