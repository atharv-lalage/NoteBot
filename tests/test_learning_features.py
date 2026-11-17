from types import SimpleNamespace

import pytest

from app.schemas import Quiz, QuizQuestion
from app.services.flashcard_service import FlashcardService
from app.services.quiz_service import QuizService
from app.services.summary_service import SummaryService


class FakeClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text=self.payload)


def test_summary_output_is_pydantic_validated() -> None:
    result = SummaryService(
        FakeClient(
            '{"title":"Systems","summary":"A review","key_topics":["Processes"],'
            '"difficulty":"beginner","study_tips":["Review scheduling"]}'
        )
    ).generate("notes")

    assert result.title == "Systems"
    assert result.difficulty == "beginner"


def test_flashcards_include_difficulty_and_source_pages() -> None:
    result = FlashcardService(
        FakeClient(
            '{"flashcards":[{"question":"What is a process?","answer":"A program in execution",'
            '"difficulty":"easy","source_pages":[2]}]}'
        )
    ).generate("notes", 1)

    assert result[0].difficulty == "easy"
    assert result[0].source_pages == [2]


def test_quiz_validates_answers_and_calculates_score() -> None:
    quiz = Quiz(
        questions=[
            QuizQuestion(
                question="Which option?",
                question_type="multiple_choice",
                options=["A", "B"],
                correct_answer="A",
                explanation="A is correct.",
                source_pages=[1],
            ),
            QuizQuestion(
                question="Name it.",
                question_type="short_answer",
                correct_answer="virtual memory",
                explanation="It manages memory.",
                source_pages=[3],
            ),
        ]
    )

    score = QuizService.score(quiz, {0: "A", 1: "Virtual   Memory"})

    assert score.score == 2
    assert score.total == 2
    assert score.percentage == 100
    assert score.correct_answers == [0, 1]


def test_quiz_rejects_invalid_multiple_choice_output() -> None:
    with pytest.raises(ValueError, match="valid options"):
        QuizService(
            FakeClient(
                '{"questions":[{"question":"Q","question_type":"multiple_choice",'
                '"options":["A","B"],"correct_answer":"C","explanation":"E",'
                '"difficulty":"easy","source_pages":[1]}]}'
            )
        ).generate("notes", 1)
