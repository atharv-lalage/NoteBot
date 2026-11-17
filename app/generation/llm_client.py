"""Isolated Gemini client and retry policy."""

import time
from collections.abc import Iterator
from typing import Any, Callable, Optional

from google import genai
from google.genai.errors import ClientError, ServerError

from app.config import settings


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured")
        self._client = genai.Client(api_key=key)

    def _call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        for attempt in range(settings.max_retries):
            try:
                return operation(*args, **kwargs)
            except (ClientError, ServerError):
                if attempt == settings.max_retries - 1:
                    raise
                time.sleep((attempt + 1) * 3)

    def generate(self, **kwargs: Any) -> Any:
        return self._call(self._client.models.generate_content, **kwargs)

    def stream(self, **kwargs: Any) -> Iterator[Any]:
        return self._call(self._client.models.generate_content_stream, **kwargs)

    def embed_content(self, **kwargs: Any) -> Any:
        return self._call(self._client.models.embed_content, **kwargs)
