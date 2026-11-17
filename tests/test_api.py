from app.api import _unique_sources
from app.schemas import SearchResult


def test_unique_sources_collapses_chunks_from_same_page() -> None:
    sources = [
        SearchResult(
            text="first",
            score=0.95,
            chunk_id="doc-page_9-chunk_0",
            document_id="doc",
            document_name="Notes",
            page_number=9,
            citation="Notes, page 9",
        ),
        SearchResult(
            text="second",
            score=0.90,
            chunk_id="doc-page_9-chunk_1",
            document_id="doc",
            document_name="Notes",
            page_number=9,
            citation="Notes, page 9",
        ),
        SearchResult(
            text="third",
            score=0.85,
            chunk_id="doc-page_11-chunk_0",
            document_id="doc",
            document_name="Notes",
            page_number=11,
            citation="Notes, page 11",
        ),
    ]

    unique = _unique_sources(sources)

    assert [source.page_number for source in unique] == [9, 11]
