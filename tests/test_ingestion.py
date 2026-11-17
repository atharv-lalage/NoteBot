from io import BytesIO

import fitz
import pytest

from app.ingestion.pdf_loader import InvalidPDFError, ScannedPDFError
from app.ingestion.chunker import ChunkingConfig, chunk_pages
from app.ingestion.pipeline import (
    DocumentRegistry,
    DuplicateDocumentError,
    process_document,
)
from app.models import DocumentStatus
from app.models import DocumentPage


def make_pdf(*texts: str) -> bytes:
    pdf = fitz.open()
    for text in texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    return pdf.tobytes()


def test_extracts_pages_and_tracks_ready_status() -> None:
    document = process_document(BytesIO(make_pdf("first page", "second page")))

    assert document.status == DocumentStatus.READY
    assert [(page.page_number, page.text) for page in document.pages or []] == [
        (1, "first page"),
        (2, "second page"),
    ]
    assert document.chunks
    assert document.chunks[0].id == (
        f"{document.document_hash}-page_1-chunk_0"
    )
    assert document.chunks[0].document_id == document.document_hash
    assert document.chunks[0].content_hash


def test_chunk_ids_are_stable_and_page_aware() -> None:
    pages = [DocumentPage(4, "HEADING\n" + ("content " * 100))]
    config = ChunkingConfig(size=400, overlap=50, strategy="heading")

    first = chunk_pages(pages, "doc-id", config)
    second = chunk_pages(pages, "doc-id", config)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.page_number == 4 for chunk in first)
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_rejects_duplicate_content_hash() -> None:
    registry = DocumentRegistry()
    data = make_pdf("same content")
    process_document(BytesIO(data), registry)

    with pytest.raises(DuplicateDocumentError):
        process_document(BytesIO(data), registry)


def test_rejects_invalid_and_scanned_pdfs() -> None:
    with pytest.raises(InvalidPDFError):
        process_document(BytesIO(b"not a pdf"))

    with pytest.raises(ScannedPDFError):
        process_document(BytesIO(make_pdf("")))
