"""FastAPI backend for the React NoteBot client."""

import io
import json
from collections.abc import Iterator
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.db_models import MessageRecord
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
    expose_headers=["X-Citations", "X-Conversation-Id"],
)


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)


class ConversationCreateRequest(BaseModel):
    document_version_id: str
    title: str = Field(default="New conversation", min_length=1, max_length=512)


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
        "document_id": str(record.document_id),
        "filename": record.filename,
        "name": record.filename,
        "status": record.status,
        "page_count": record.page_count,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "error": record.error,
        "version_number": record.version_number,
        "version_group_id": str(record.document_id),
        "is_current": record.is_current,
        "preview_url": f"/api/document-versions/{record.id}/file",
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


def _conversation_response(conversation: Any) -> dict[str, Any]:
    version = conversation.version
    return {
        "id": str(conversation.id),
        "document_id": str(version.document_id),
        "document_version_id": str(version.id),
        "title": conversation.title,
        "version_number": version.version_number,
        "created_at": conversation.created_at.isoformat()
        if conversation.created_at
        else None,
    }


def _message_response(message: Any) -> dict[str, Any]:
    document_name = message.conversation.version.document.name
    version_number = message.conversation.version.version_number
    citations = [
        {
            "chunk_id": source.chunk_id,
            "document_id": str(source.document_id),
            "document_version_id": str(source.document_version_id),
            "document_name": document_name,
            "version_number": source.document_version or version_number,
            "page_number": source.page_number,
            "score": source.score,
            "citation": (
                f"{document_name}, Version "
                f"{source.document_version or version_number}, Page {source.page_number}"
            ),
        }
        for source in message.sources
    ]
    return {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "role": message.role,
        "content": message.content,
        "status": message.status,
        "created_at": message.created_at.isoformat() if message.created_at else None,
        "completed_at": message.completed_at.isoformat()
        if message.completed_at
        else None,
        "sources": citations,
        "citations": citations,
    }


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


@app.post("/api/conversations", status_code=201)
def create_conversation(request: ConversationCreateRequest) -> dict[str, Any]:
    try:
        with SessionLocal() as session:
            conversation = DocumentRepository(session).create_conversation(
                request.document_version_id,
                title=request.title,
            )
            return _conversation_response(conversation)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/conversations")
def list_conversations(
    document_version_id: Optional[str] = Query(default=None),
) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as session:
            return [
                _conversation_response(conversation)
                for conversation in DocumentRepository(session).list_conversations(
                    document_version_id
                )
            ]
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        repository = DocumentRepository(session)
        conversation = repository.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation was not found.")
        response = _conversation_response(conversation)
        response["messages"] = [
            _message_response(message)
            for message in repository.list_messages(conversation.id)
        ]
        return response


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    try:
        with SessionLocal() as session:
            DocumentRepository(session).delete_conversation(conversation_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        return [_document_response(record) for record in DocumentRepository(session).list_documents()]


@app.get("/api/documents/{document_id}/versions")
def list_document_versions(document_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        try:
            versions = DocumentRepository(session).list_versions(document_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if not versions:
            raise HTTPException(status_code=404, detail="Document was not found.")
        return [_document_response(version) for version in versions]


def _asset_response(asset: Any) -> dict[str, Any]:
    version = asset.version
    return {
        "id": str(asset.id),
        "asset_type": asset.asset_type,
        "payload": asset.payload,
        "document_id": str(asset.document_id),
        "document_version_id": str(asset.document_version_id),
        "document_name": version.document.name,
        "version_number": version.version_number,
        "created_at": asset.created_at.isoformat() if asset.created_at else None,
    }


@app.get("/api/documents/{document_id}/assets")
def list_document_assets(document_id: str) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as session:
            repository = DocumentRepository(session)
            assets = repository.list_assets(document_id)
            if not assets:
                versions = repository.list_versions(document_id)
                if not versions:
                    raise ValueError("Document version was not found")
            return [_asset_response(asset) for asset in assets]
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


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
                # Older versions predate PDF byte storage. Reusing the same
                # hash is safe and lets those versions gain a preview without
                # creating a new immutable version.
                if not existing.pdf_data:
                    existing.pdf_data = data
                    session.commit()
                return JSONResponse(
                    status_code=200,
                    content=_document_response(existing),
                )
        document = process_document(io.BytesIO(data))
        document.filename = file.filename or "uploaded.pdf"
        document.pdf_data = data
        with SessionLocal() as session:
            repository = DocumentRepository(session)
            record = repository.create_document_version(document)
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


@app.get("/api/document-versions/{version_id}/file")
def get_document_file(version_id: str) -> Response:
    with SessionLocal() as session:
        record = DocumentRepository(session).get_version(version_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Document version was not found.")
        if not record.pdf_data:
            raise HTTPException(
                status_code=404,
                detail="This document version does not have a preview file.",
            )
        return Response(
            content=record.pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{record.filename}"',
                "Cache-Control": "private, max-age=3600",
            },
        )


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    try:
        with SessionLocal() as session:
            DocumentRepository(session).delete(document_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/documents/{document_id}/chat")
def chat(document_id: str, request: ChatRequest) -> StreamingResponse:
    with SessionLocal() as session:
        repository = DocumentRepository(session)
        conversation = repository.get_conversation(request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation was not found.")
        requested_version = repository.get_version(document_id)
        if requested_version is None:
            raise HTTPException(status_code=404, detail="Document version was not found.")
        if requested_version.document_id != conversation.document_id:
            raise HTTPException(
                status_code=409,
                detail="Conversation is not pinned to the requested document.",
            )
        pinned_id = str(conversation.document_version_id)
        try:
            conversation = repository.create_conversation(pinned_id, request.conversation_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        history = repository.get_conversation_history(conversation.id)
        repository.add_message(
            conversation.id,
            "user",
            request.question,
            status="completed",
        )
        conversation_id = str(conversation.id)
        pinned_document = repository.get_ready(pinned_id)
        if pinned_document is None:
            raise HTTPException(status_code=409, detail="Document is not ready.")
        document_version = pinned_document.version_number
    retriever, client, _ = _services(pinned_id)
    answer_service = AnswerService(client)
    if answer_service.classify_intent(request.question, history) == "CASUAL":
        with SessionLocal() as session:
            assistant_message = DocumentRepository(session).add_message(
                conversation_id,
                "assistant",
                "",
                document_version=document_version,
                status="pending",
            )
        assistant_message_id = str(assistant_message.id)

        def casual_response() -> Iterator[str]:
            emitted = ""
            failed = False
            completed = False
            try:
                for token in answer_service.stream_casual(request.question, history):
                    emitted += token
                    yield token
                completed = True
            except RuntimeError as error:
                failed = True
                emitted = str(error)
                yield f"\n\n{error}\n"
                with SessionLocal() as session:
                    DocumentRepository(session).update_message(
                        assistant_message_id,
                        content=emitted,
                        status="failed",
                    )
            finally:
                if completed:
                    with SessionLocal() as session:
                        DocumentRepository(session).update_message(
                            assistant_message_id,
                            content=emitted,
                            status="completed",
                        )
                elif not failed:
                    with SessionLocal() as session:
                        DocumentRepository(session).update_message(
                            assistant_message_id,
                            content=emitted or "Response stream interrupted.",
                            status="failed",
                        )

        return StreamingResponse(
            casual_response(),
            media_type="text/plain",
            headers={
                "X-Source-Count": "0",
                "X-Citations": "[]",
                "X-Conversation-Id": conversation_id,
            },
        )
    search_query = answer_service.rewrite_query(request.question, history)
    sources = retriever.retrieve(
        search_query,
        version_id=pinned_id,
        k=settings.retrieval_top_k,
    )
    citation_sources = _unique_sources(sources)

    def generate() -> Iterator[str]:
        if not sources:
            with SessionLocal() as session:
                message = DocumentRepository(session).add_message(
                    conversation_id,
                    "assistant",
                    REFUSAL_MESSAGE,
                    document_version=document_version,
                    status="completed",
                )
            yield REFUSAL_MESSAGE
            return
        with SessionLocal() as session:
            assistant_message = DocumentRepository(session).add_message(
                conversation_id,
                "assistant",
                "",
                document_version=document_version,
                status="pending",
            )
        assistant_message_id = str(assistant_message.id)
        emitted = ""
        completed = False
        failed = False
        try:
            for token in answer_service.stream_answer(request.question, sources, history):
                emitted += token
                yield token
            completed = True
        except RuntimeError as error:
            failed = True
            with SessionLocal() as session:
                DocumentRepository(session).update_message(
                    assistant_message_id,
                    content=str(error),
                    status="failed",
                )
            yield f"\n\n{error}\n"
            return
        finally:
            if not completed and not failed:
                with SessionLocal() as session:
                    DocumentRepository(session).update_message(
                        assistant_message_id,
                        content=emitted or "Response stream interrupted.",
                        status="failed",
                    )
        validated = answer_service.validate_answer(emitted)
        with SessionLocal() as session:
            repository = DocumentRepository(session)
            repository.update_message(
                assistant_message_id,
                content=validated,
                status="completed",
            )
            if validated != REFUSAL_MESSAGE:
                message = session.get(MessageRecord, assistant_message.id)
                if message is not None:
                    repository.add_message_sources(message, sources, document_version)
                    session.commit()

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
                        "document_version_id": source.document_version_id,
                        "document_name": source.document_name,
                        "version_number": source.version_number,
                        "page_number": source.page_number,
                        "score": source.score,
                        "citation": source.citation,
                    }
                    for source in citation_sources
                ]
            ),
            "X-Conversation-Id": conversation_id,
        },
    )


@app.post("/api/conversations/{conversation_id}/messages")
def send_conversation_message(
    conversation_id: str, request: ChatRequest
) -> StreamingResponse:
    if request.conversation_id != conversation_id:
        raise HTTPException(status_code=409, detail="Conversation ID mismatch.")
    with SessionLocal() as session:
        conversation = DocumentRepository(session).get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation was not found.")
        version_id = str(conversation.document_version_id)
    return chat(version_id, request)


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
