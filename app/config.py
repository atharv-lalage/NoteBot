"""Centralized application configuration."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    chat_model: str = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
    embedding_model: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://notebot:notebot@localhost:5432/notebot",
    )
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
    retrieval_candidate_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "16"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    retrieval_similarity_threshold: float = float(
        os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.35")
    )
    retrieval_neighbor_window: int = int(os.getenv("RETRIEVAL_NEIGHBOR_WINDOW", "1"))
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")


settings = Settings()
