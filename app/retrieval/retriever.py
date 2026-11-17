"""Query-to-document retrieval orchestration."""

from typing import Optional

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.pgvector_store import PgVectorStore
from app.schemas import SearchResult
from app.config import settings


class Retriever:
    def __init__(self, embeddings: EmbeddingService, store: PgVectorStore):
        self.embeddings = embeddings
        self.store = store

    def retrieve(
        self, query: str, k: int = settings.retrieval_top_k, document_ids: Optional[list[str]] = None
    ) -> list[SearchResult]:
        query_embedding = self.embeddings.embed([query], task="RETRIEVAL_QUERY")[0]
        return self.store.search(
            query_embedding,
            document_ids=document_ids,
            top_k=min(k, self.store.candidate_k),
        )
