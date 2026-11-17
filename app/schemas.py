"""Shared data contracts used across application layers."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    chunk_id: str = ""
    page_number: int = 0
    document_id: str = ""
    document_version_id: str = ""
    document_name: str = ""
    version_number: int = 0
    citation: str = ""


@dataclass
class ChatResponse:
    content: str
    sources: list[SearchResult] = field(default_factory=list)


class Flashcard(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source_pages: list[int] = Field(default_factory=list, min_length=1)


class Summary(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_topics: list[str] = Field(min_length=1)
    difficulty: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    study_tips: list[str] = Field(min_length=1)


class QuizQuestion(BaseModel):
    question: str = Field(min_length=1)
    question_type: Literal["multiple_choice", "short_answer"]
    options: list[str] = Field(default_factory=list)
    correct_answer: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source_pages: list[int] = Field(default_factory=list, min_length=1)


class Quiz(BaseModel):
    questions: list[QuizQuestion] = Field(min_length=1)


class QuizSubmission(BaseModel):
    answers: dict[int, str]


class QuizScore(BaseModel):
    score: int = Field(ge=0)
    total: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)
    correct_answers: list[int]


JsonObject = dict[str, Any]
