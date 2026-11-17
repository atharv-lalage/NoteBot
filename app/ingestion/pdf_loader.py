"""Validated, page-aware PDF text extraction."""

from io import BytesIO
from typing import BinaryIO

import fitz

from app.config import settings
from app.models import DocumentPage


class PDFIngestionError(ValueError):
    """Base class for user-correctable PDF upload errors."""


class InvalidPDFError(PDFIngestionError):
    """Raised when an upload is not a readable PDF."""


class FileTooLargeError(PDFIngestionError):
    """Raised when an upload exceeds the configured size limit."""


class ScannedPDFError(PDFIngestionError):
    """Raised when a PDF has no extractable text."""


def read_file_bytes(file: BinaryIO) -> bytes:
    data = file.read()
    if not data:
        raise InvalidPDFError("The uploaded file is empty.")
    if len(data) > settings.max_file_size_mb * 1024 * 1024:
        raise FileTooLargeError(
            f"File is too large. Maximum allowed size is {settings.max_file_size_mb} MB."
        )
    return data


def extract_pages(data: bytes) -> list[DocumentPage]:
    if not data.startswith(b"%PDF-"):
        raise InvalidPDFError("Only valid PDF files can be uploaded.")
    try:
        pdf = fitz.open(stream=BytesIO(data), filetype="pdf")
    except (fitz.FileDataError, ValueError) as error:
        raise InvalidPDFError("The uploaded file could not be opened as a PDF.") from error
    try:
        pages = [
            DocumentPage(page_number=index + 1, text=page.get_text("text").strip())
            for index, page in enumerate(pdf)
        ]
    finally:
        pdf.close()
    if not pages or not any(page.text for page in pages):
        raise ScannedPDFError(
            "No selectable text was found. This may be a scanned PDF; upload a text-based PDF."
        )
    return pages


def extract_text(file: BinaryIO) -> str:
    """Backward-compatible helper returning all extracted text."""
    return "\n\n".join(page.text for page in extract_pages(read_file_bytes(file)))
