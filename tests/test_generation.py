from types import SimpleNamespace

from app.generation.answer_service import AnswerService
from app.generation.prompts import (
    GROUNDING_SYSTEM_INSTRUCTION,
    REFUSAL_MESSAGE,
    chat_prompt,
    format_context,
)
from app.schemas import SearchResult


class FakeClient:
    def __init__(self, text: str = "rewritten query") -> None:
        self.text = text

    def generate(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text=self.text)

    def stream(self, **kwargs: object) -> list[SimpleNamespace]:
        return [SimpleNamespace(text="The answer is supported.")]


def source() -> SearchResult:
    return SearchResult(
        text="Operating systems manage computer resources.",
        score=0.91,
        chunk_id="doc-page_12-chunk_0",
        page_number=12,
        document_name="Operating Systems",
        citation="Operating Systems, page 12",
    )


def test_empty_retrieval_returns_exact_refusal() -> None:
    answer = AnswerService(FakeClient()).answer("Who wrote this?", [])

    assert answer.content == REFUSAL_MESSAGE
    assert answer.sources == []


def test_grounded_prompt_separates_document_content() -> None:
    prompt = chat_prompt(format_context([source()]), "", "What do operating systems manage?")

    assert "DOCUMENT EXCERPTS" in prompt
    assert "Operating systems manage computer resources." in prompt
    assert "doc-page_12-chunk_0" not in prompt
    assert "Never include" in GROUNDING_SYSTEM_INSTRUCTION
    assert "The answer is" in prompt


def test_query_rewriting_uses_conversation_history() -> None:
    answer_service = AnswerService(FakeClient("what does the second concept mean?"))

    assert answer_service.rewrite_query("What does the second one mean?", "USER: Explain two concepts.") == (
        "what does the second concept mean?"
    )
