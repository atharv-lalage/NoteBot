"""Persistent pgvector retrieval with filtering and citation metadata."""

from collections.abc import Sequence
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db_models import ChunkRecord, DocumentVersionRecord
from app.schemas import SearchResult


def _result(record: ChunkRecord, score: float) -> SearchResult:
    document_id = str(getattr(record, "document_id", ""))
    document_version_id = str(getattr(record, "document_version_id", document_id))
    version = getattr(record, "version", None)
    document_name = version.document.name if version and version.document else document_id
    version_number = version.version_number if version else 0
    citation = (
        f"{document_name}, Version {version_number}, Page {record.page_number}"
        if version_number
        else f"{document_name}, page {record.page_number}"
    )
    return SearchResult(
        text=record.content,
        score=score,
        chunk_id=record.id,
        page_number=record.page_number,
        document_id=document_id,
        document_version_id=document_version_id,
        document_name=document_name,
        version_number=version_number,
        citation=citation,
    )


class PgVectorStore:
    """Runs the production retrieval pipeline against PostgreSQL."""

    def __init__(
        self,
        session: Session,
        *,
        candidate_k: int = settings.retrieval_candidate_k,
        top_k: int = settings.retrieval_top_k,
        similarity_threshold: float = settings.retrieval_similarity_threshold,
        neighbor_window: int = settings.retrieval_neighbor_window,
    ):
        if candidate_k < 1 or top_k < 1 or top_k > candidate_k:
            raise ValueError("top_k must be positive and no greater than candidate_k")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")
        self.session = session
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.neighbor_window = neighbor_window

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        version_id: str,
        top_k: Optional[int] = None,
        query_text: Optional[str] = None,
    ) -> list[SearchResult]:
        """Return thresholded, citation-ready results from filtered candidates."""
        distance = ChunkRecord.embedding.cosine_distance(list(query_embedding))
        filters = [
            ChunkRecord.embedding.is_not(None),
            DocumentVersionRecord.status == "ready",
        ]
        filters.append(
            ChunkRecord.document_version_id == UUID(str(version_id))
        )

        candidate_rows = self.session.execute(
            select(ChunkRecord, (1 - distance).label("similarity"))
            .join(
                DocumentVersionRecord,
                DocumentVersionRecord.id == ChunkRecord.document_version_id,
            )
            .where(and_(*filters))
            .order_by(distance)
            .limit(self.candidate_k)
        ).all()

        candidates = [
            (record, float(score))
            for record, score in candidate_rows
            if float(score) >= self.similarity_threshold
        ]
        if not candidates:
            return []

        selected: dict[str, tuple[ChunkRecord, float]] = {
            record.id: (record, score) for record, score in candidates
        }
        lexical_terms = {
            term.strip("?!.,:;()[]{}\"'").lower()
            for term in (query_text or "").split()
            if len(term.strip("?!.,:;()[]{}\"'")) >= 4
        }
        lexical_terms.difference_update(
            {"what", "which", "where", "when", "that", "this", "from", "with", "does"}
        )
        if lexical_terms:
            lexical_matches = self.session.scalars(
                select(ChunkRecord)
                .where(
                    and_(
                        ChunkRecord.document_version_id == UUID(str(version_id)),
                        ChunkRecord.embedding.is_not(None),
                        or_(
                            *[
                                ChunkRecord.content.ilike(f"%{term}%")
                                for term in lexical_terms
                            ]
                        ),
                    )
                )
            ).all()
            lexical_score = max(score for _, score in candidates) + 0.001
            selected.update(
                {
                    record.id: (record, lexical_score)
                    for record in lexical_matches
                }
            )
        if self.neighbor_window:
            neighbor_conditions = []
            for record, score in candidates:
                neighbor_conditions.append(
                    and_(
                        ChunkRecord.document_version_id == record.document_version_id,
                        ChunkRecord.page_number == record.page_number,
                        ChunkRecord.chunk_index >= record.chunk_index - self.neighbor_window,
                        ChunkRecord.chunk_index <= record.chunk_index + self.neighbor_window,
                    )
                )
            neighbors = self.session.scalars(
                select(ChunkRecord)
                .where(
                    and_(
                        ChunkRecord.embedding.is_not(None),
                        or_(*neighbor_conditions),
                    )
                )
            ).all()
            candidate_scores = {record.id: score for record, score in candidates}
            for record in neighbors:
                if record.id not in selected:
                    selected[record.id] = (record, candidate_scores.get(record.id, 0.0))

        ranked = sorted(selected.values(), key=lambda item: item[1], reverse=True)
        limit = top_k if top_k is not None else self.top_k
        return [_result(record, score) for record, score in ranked[:limit]]
