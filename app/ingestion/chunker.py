"""Metadata-rich text chunking utilities."""

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.models import DocumentChunk, DocumentPage


@dataclass(frozen=True)
class ChunkingConfig:
    size: int = settings.chunk_size
    overlap: int = settings.chunk_overlap
    strategy: str = "recursive"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")
        if self.strategy not in {"recursive", "heading"}:
            raise ValueError("Chunking strategy must be 'recursive' or 'heading'")


def _recursive_split(text: str, config: ChunkingConfig) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.size,
        chunk_overlap=config.overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _heading_split(text: str, config: ChunkingConfig) -> List[str]:
    """Split at likely headings first, then apply recursive splitting per section."""
    sections: List[str] = []
    current: List[str] = []
    heading_pattern = re.compile(r"^(?:#{1,6}\s+|[A-Z][A-Z0-9\s:,-]{3,})$")
    for line in text.splitlines():
        if line.strip() and heading_pattern.match(line.strip()) and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [chunk for section in sections for chunk in _recursive_split(section, config)]


def split_text(text: str, config: ChunkingConfig = ChunkingConfig()) -> List[str]:
    splitter: Callable[[str, ChunkingConfig], List[str]] = (
        _heading_split if config.strategy == "heading" else _recursive_split
    )
    return splitter(text, config)


def chunk_text(text: str) -> list[str]:
    return split_text(text, ChunkingConfig())


def chunk_pages(
    pages: list[DocumentPage],
    document_id: str = "document",
    config: ChunkingConfig = ChunkingConfig(),
) -> list[DocumentChunk]:
    """Chunk pages independently and attach deterministic metadata."""
    chunks: list[DocumentChunk] = []
    for page in pages:
        for chunk_index, content in enumerate(split_text(page.text, config)):
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}-page_{page.page_number}-chunk_{chunk_index}",
                    document_id=document_id,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    content=content,
                    content_hash=content_hash,
                )
            )
    return chunks
