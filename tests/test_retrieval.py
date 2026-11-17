from types import SimpleNamespace

from app.retrieval.pgvector_store import _result


def test_retrieval_result_contains_citation_metadata() -> None:
    record = SimpleNamespace(
        id="doc-page_4-chunk_2",
        document_id="document-id",
        page_number=4,
        content="Relevant passage",
    )

    result = _result(record, 0.82)

    assert result.chunk_id == "doc-page_4-chunk_2"
    assert result.document_id == "document-id"
    assert result.page_number == 4
    assert result.score == 0.82
    assert "page 4" in result.citation
