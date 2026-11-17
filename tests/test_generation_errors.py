from types import SimpleNamespace

from google.genai.errors import ServerError

from app.generation.answer_service import AnswerService
from app.schemas import SearchResult


class UnavailableClient:
    def stream(self, **kwargs: object) -> list[SimpleNamespace]:
        def failing_stream() -> object:
            raise ServerError(503, {"error": {"message": "temporarily unavailable"}})
            yield  # pragma: no cover

        return failing_stream()


def test_model_stream_failure_is_reported_to_caller() -> None:
    source = SearchResult(text="Evidence", score=0.9, page_number=1)

    try:
        list(AnswerService(UnavailableClient()).stream_answer("Question", [source]))
    except RuntimeError as error:
        assert str(error) == "The Gemini model is temporarily unavailable. Please try again in a moment."
    else:
        raise AssertionError("Expected a user-facing model availability error")
