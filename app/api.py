"""FastAPI backend for the React NoteBot client."""

import io
import json
from collections.abc import Iterator
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.generation.answer_service import AnswerService
from app.generation.llm_client import GeminiClient
from app.generation.prompts import REFUSAL_MESSAGE
from app.config import settings
from app.ingestion.pipeline import (
    DuplicateDocumentError,
    calculate_document_hash,
    process_document,
)
from app.repositories import DocumentRepository
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.retriever import Retriever
from app.services.flashcard_service import FlashcardService
from app.services.quiz_service import QuizService
from app.services.summary_service import SummaryService
from app.schemas import Quiz, QuizScore, QuizSubmission
from app.schemas import SearchResult

app = FastAPI(title="NoteBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Citations"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: str = Field(default="", max_length=20000)
    document_ids: Optional[list[str]] = None


class FlashcardRequest(BaseModel):
    count: int = Field(default=5, ge=3, le=15)


class QuizRequest(BaseModel):
    count: int = Field(default=5, ge=3, le=15)


class QuizSubmissionRequest(BaseModel):
    quiz: Quiz
    answers: dict[int, str]


def _document_response(record: Any) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "filename": record.filename,
        "status": record.status,
        "page_count": record.page_count,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "error": record.error,
    }


def _unique_sources(sources: list[SearchResult]) -> list[SearchResult]:
    """Collapse multiple retrieved chunks from the same document page."""
    unique: list[SearchResult] = []
    seen: set[tuple[str, int]] = set()
    for source in sources:
        key = (source.document_id, source.page_number)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _services(document_id: str) -> tuple[Retriever, GeminiClient, list[str]]:
    with SessionLocal() as session:
        repository = DocumentRepository(session)
        if repository.get_ready(document_id) is None:
            raise HTTPException(status_code=409, detail="Document is not ready.")
        client = GeminiClient()
        retriever = Retriever(EmbeddingService(client), PgVectorStore(session))
        notes = repository.get_chunk_texts(document_id)
        return retriever, client, notes


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        return [_document_response(record) for record in DocumentRepository(session).list_documents()]


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    data = await file.read()
    document_hash = calculate_document_hash(data)
    try:
        with SessionLocal() as session:
            repository = DocumentRepository(session)
            existing = repository.find_by_hash(document_hash)
            if existing:
                raise DuplicateDocumentError(
                    f"This document was already uploaded as '{existing.filename}'."
                )
        document = process_document(io.BytesIO(data))
        document.filename = file.filename or "uploaded.pdf"
        with SessionLocal() as session:
            repository = DocumentRepository(session)
            record = repository.create_processing(document)
        try:
            client = GeminiClient()
            chunks = document.chunks or []
            embeddings = EmbeddingService(client).embed([chunk.content for chunk in chunks])
            with SessionLocal() as session:
                record = DocumentRepository(session).complete(record.id, embeddings)
            return _document_response(record)
        except Exception as error:
            with SessionLocal() as session:
                DocumentRepository(session).fail(record.id, str(error))
            raise
    except DuplicateDocumentError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {error}") from error


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    try:
        with SessionLocal() as session:
            DocumentRepository(session).delete(document_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/documents/{document_id}/chat")
def chat(document_id: str, request: ChatRequest) -> StreamingResponse:
    retriever, client, _ = _services(document_id)
    answer_service = AnswerService(client)
    if answer_service.classify_intent(request.question, request.history) == "CASUAL":
        return StreamingResponse(
            answer_service.stream_casual(request.question, request.history),
            media_type="text/plain",
            headers={"X-Source-Count": "0", "X-Citations": "[]"},
        )
    scope = request.document_ids or [document_id]
    search_query = answer_service.rewrite_query(request.question, request.history)
    sources = retriever.retrieve(
        search_query,
        k=settings.retrieval_top_k,
        document_ids=scope,
    )
    citation_sources = _unique_sources(sources)

    def generate() -> Iterator[str]:
        if not sources:
            yield REFUSAL_MESSAGE
            return
        emitted = ""
        try:
            for token in answer_service.stream_answer(request.question, sources, request.history):
                emitted += token
                yield token
        except RuntimeError as error:
            yield f"\n\n{error}\n"
            return
        validated = answer_service.validate_answer(emitted)
        if validated == REFUSAL_MESSAGE:
            return

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Source-Count": str(len(sources)),
            "X-Citations": json.dumps(
                [
                    {
                        "chunk_id": source.chunk_id,
                        "document_id": source.document_id,
                        "document_name": source.document_name,
                        "page_number": source.page_number,
                        "score": source.score,
                        "citation": source.citation,
                    }
                    for source in citation_sources
                ]
            ),
        },
    )


@app.post("/api/documents/{document_id}/summary")
def summary(document_id: str) -> Any:
    _, client, notes = _services(document_id)
    try:
        result = SummaryService(client).generate("\n\n".join(notes[:20]))
    except ValueError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    with SessionLocal() as session:
        DocumentRepository(session).save_asset(document_id, "summary", result.model_dump())
    return result.model_dump()


@app.post("/api/documents/{document_id}/flashcards")
def flashcards(document_id: str, request: FlashcardRequest) -> list[dict[str, Any]]:
    _, client, notes = _services(document_id)
    try:
        cards = FlashcardService(client).generate("\n\n".join(notes[:20]), request.count)
    except ValueError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    with SessionLocal() as session:
        DocumentRepository(session).save_asset(
            document_id,
            "flashcards",
            {"flashcards": [card.model_dump() for card in cards]},
        )
    return [card.model_dump() for card in cards]


@app.post("/api/documents/{document_id}/quiz", response_model=Quiz)
def quiz(document_id: str, request: QuizRequest) -> Quiz:
    _, client, notes = _services(document_id)
    try:
        result = QuizService(client).generate("\n\n".join(notes[:20]), request.count)
    except ValueError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    with SessionLocal() as session:
        DocumentRepository(session).save_asset(document_id, "quiz", result.model_dump())
    return result


@app.post("/api/documents/{document_id}/quiz/score", response_model=QuizScore)
def score_quiz(document_id: str, request: QuizSubmissionRequest) -> QuizScore:
    with SessionLocal() as session:
        if DocumentRepository(session).get_ready(document_id) is None:
            raise HTTPException(status_code=409, detail="Document is not ready.")
    return QuizService.score(request.quiz, request.answers)
