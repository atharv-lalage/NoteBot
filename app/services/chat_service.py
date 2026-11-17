"""Chat use case."""

from collections.abc import Iterator

from app.generation.answer_service import AnswerService
from app.retrieval.retriever import Retriever
from app.schemas import ChatResponse, SearchResult


class ChatService:
    def __init__(self, retriever: Retriever, answer_service: AnswerService):
        self.retriever = retriever
        self.answer_service = answer_service

    def answer(
        self, question: str, *, version_id: str, history: str = "", k: int = 4
    ) -> ChatResponse:
        sources = self.retriever.retrieve(question, version_id=version_id, k=k)
        return self.answer_service.answer(question, sources, history)

    def stream_answer(
        self, question: str, *, version_id: str, history: str = "", k: int = 4
    ) -> tuple[list[SearchResult], Iterator[str]]:
        sources = self.retriever.retrieve(question, version_id=version_id, k=k)
        return sources, self.answer_service.stream_answer(question, sources, history)
