"""Summary use case."""

from google.genai import types

from app.config import settings
from app.generation.llm_client import GeminiClient
from app.schemas import Summary
from app.services.structured_output import parse_json_object, response_schema


class SummaryService:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(self, notes: str) -> Summary:
        prompt = f"""Analyze the notes below and return exactly one JSON object with these fields:
title (string), summary (string), key_topics (array of strings), difficulty (one of
beginner, intermediate, advanced), and study_tips (array of strings). Do not include
any other fields or markdown. NOTES:
{notes}"""
        response = self.client.generate(
            model=settings.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema(Summary),
            ),
        )
        return parse_json_object(response.text or "", Summary)
