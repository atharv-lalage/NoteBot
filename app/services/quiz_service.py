"""Quiz generation and scoring use case."""

import re
from typing import Iterable

from google.genai import types

from app.config import settings
from app.generation.llm_client import GeminiClient
from app.schemas import Quiz, QuizQuestion, QuizScore
from app.services.structured_output import parse_json_object, response_schema


class QuizService:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(self, notes: str, count: int = 5) -> Quiz:
        prompt = (
            f"Generate exactly {count} quiz questions from these notes as JSON with a "
            "questions array. Mix multiple_choice and short_answer questions. Each "
            "question must include question, question_type, options, correct_answer, "
            "explanation, difficulty, and source_pages. For short_answer, use an empty "
            "options array. source_pages must contain only integer page numbers from "
            "the [Document page N] markers, never emails or other text. Every correct "
            "answer must be supported by the notes. NOTES:\n"
            f"{notes}"
        )
        response = self.client.generate(
            model=settings.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema(Quiz),
            ),
        )
        quiz = parse_json_object(response.text or "", Quiz)
        if len(quiz.questions) != count:
            raise ValueError(f"Generated quiz must contain exactly {count} questions")
        for question in quiz.questions:
            if question.question_type == "multiple_choice":
                if len(question.options) < 2 or question.correct_answer not in question.options:
                    raise ValueError("Multiple-choice questions need valid options and an answer")
            elif question.options:
                raise ValueError("Short-answer questions must not include options")
        return quiz

    @staticmethod
    def score(quiz: Quiz, answers: dict[int, str]) -> QuizScore:
        correct: list[int] = []
        for index, question in enumerate(quiz.questions):
            submitted = answers.get(index, "")
            if _answer_matches(question, submitted):
                correct.append(index)
        total = len(quiz.questions)
        points = len(correct)
        return QuizScore(
            score=points,
            total=total,
            percentage=round(points / total * 100, 2) if total else 0,
            correct_answers=correct,
        )


def _answer_matches(question: QuizQuestion, submitted: str) -> bool:
    if question.question_type == "multiple_choice":
        return submitted.strip() == question.correct_answer.strip()
    normalize = lambda value: re.sub(r"\s+", " ", value.strip().lower())
    return normalize(submitted) == normalize(question.correct_answer)
