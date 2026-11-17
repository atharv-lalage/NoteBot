"""Flashcard generation use case."""

from google.genai import types

from app.config import settings
from app.generation.llm_client import GeminiClient
from app.schemas import Flashcard
from app.services.structured_output import parse_json_object, parse_json_value, response_schema
from pydantic import BaseModel, Field


class FlashcardSet(BaseModel):
    flashcards: list[Flashcard] = Field(min_length=1)


class FlashcardService:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(self, notes: str, count: int = 5) -> list[Flashcard]:
        response = self.client.generate(
            model=settings.chat_model,
            contents=f"""Generate exactly {count} flashcards from these notes.
Return one JSON object with a 'flashcards' array. Each card must include question,
answer, difficulty, and source_pages. source_pages must contain only integer page
numbers present in the [Document page N] markers. Do not include email addresses,
authors, or any other values in source_pages. NOTES:
{notes}""",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema(FlashcardSet),
            ),
        )
        text = response.text or ""
        try:
            return parse_json_object(text, FlashcardSet).flashcards
        except ValueError:
            return parse_json_value(text, list[Flashcard])
