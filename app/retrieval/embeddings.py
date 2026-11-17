"""Gemini embedding adapter."""

from typing import Any

import numpy as np
from google.genai import types

from app.config import settings
from app.generation.llm_client import GeminiClient


class EmbeddingService:
    def __init__(self, client: GeminiClient):
        self.client = client

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            result: Any = self.client.embed_content(
                model=settings.embedding_model,
                contents=texts[start : start + 100],
                config=types.EmbedContentConfig(task_type=task),
            )
            vectors.extend(item.values for item in result.embeddings)
        return np.asarray(vectors)
