"""In-memory cosine-similarity vector store."""

import numpy as np

from app.models import DocumentChunk
from app.schemas import SearchResult


class InMemoryVectorStore:
    def __init__(self, chunks: list[DocumentChunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have a corresponding embedding")
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query_embedding: np.ndarray, k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        scores = self.embeddings @ query_embedding / np.where(norms == 0, 1, norms)
        indices = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(
                text=self.chunks[i].content,
                score=float(scores[i]),
                chunk_id=self.chunks[i].id,
                page_number=self.chunks[i].page_number,
            )
            for i in indices
        ]
