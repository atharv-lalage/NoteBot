"""Answer generation from retrieved context."""

from collections.abc import Iterator

from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.config import settings
from app.generation.llm_client import GeminiClient
from app.generation.prompts import (
    CASUAL_SYSTEM_INSTRUCTION,
    GROUNDING_SYSTEM_INSTRUCTION,
    INTENT_SYSTEM_INSTRUCTION,
    REFUSAL_MESSAGE,
    casual_prompt,
    chat_prompt,
    format_context,
    intent_prompt,
    query_rewrite_prompt,
    QUERY_REWRITE_SYSTEM_INSTRUCTION,
)
from app.schemas import ChatResponse, SearchResult


class AnswerService:
    def __init__(self, client: GeminiClient):
        self.client = client

    def rewrite_query(self, question: str, history: str = "") -> str:
        if not history.strip():
            return question.strip()
        response = self.client.generate(
            model=settings.chat_model,
            contents=query_rewrite_prompt(question, history),
            config=types.GenerateContentConfig(
                system_instruction=QUERY_REWRITE_SYSTEM_INSTRUCTION,
            ),
        )
        rewritten = (response.text or "").strip()
        if not rewritten:
            raise ValueError("Query rewriting returned an empty query")
        return rewritten

    def classify_intent(self, question: str, history: str = "") -> str:
        response = self.client.generate(
            model=settings.chat_model,
            contents=intent_prompt(question, history),
            config=types.GenerateContentConfig(
                system_instruction=INTENT_SYSTEM_INSTRUCTION,
            ),
        )
        return "CASUAL" if (response.text or "").strip().upper() == "CASUAL" else "NOTES"

    def stream_casual(self, question: str, history: str = "") -> Iterator[str]:
        try:
            response = self.client.stream(
                model=settings.chat_model,
                contents=casual_prompt(question, history),
                config=types.GenerateContentConfig(
                    system_instruction=CASUAL_SYSTEM_INSTRUCTION,
                ),
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except (ClientError, ServerError) as error:
            raise RuntimeError(
                "The Gemini model is temporarily unavailable. Please try again in a moment."
            ) from error

    def stream_answer(self, question: str, sources: list[SearchResult], history: str = "") -> Iterator[str]:
        if not sources:
            yield REFUSAL_MESSAGE
            return
        try:
            response = self.client.stream(
                model=settings.chat_model,
                contents=chat_prompt(format_context(sources), history, question),
                config=types.GenerateContentConfig(
                    system_instruction=GROUNDING_SYSTEM_INSTRUCTION,
                ),
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except (ClientError, ServerError) as error:
            raise RuntimeError(
                "The Gemini model is temporarily unavailable. Please try again in a moment."
            ) from error

    def answer(self, question: str, sources: list[SearchResult], history: str = "") -> ChatResponse:
        content = self.validate_answer("".join(self.stream_answer(question, sources, history)))
        return ChatResponse(content, sources)
    @staticmethod
    def validate_answer(content: str) -> str:
        normalized = content.strip()
        if not normalized or normalized.lower().startswith("i could not find"):
            return REFUSAL_MESSAGE
        return normalized
