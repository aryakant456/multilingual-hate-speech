"""Request / response schemas of the API (Phase 14).

POST /predict
  request : {"text": "...", "explain": true, "details": false}
  response: {"prediction": "HATE", "confidence": 0.92, "language": "English", "important_words": ["hate", "people"]}
            (+ word-level highlights, method info and timings when "details": true)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["I hate those disgusting people"])
    explain: bool = Field(True, description="Compute the important words (slower). False = prediction only (fast).")
    details: bool = Field(False, description="Also return word-level highlights, method info, warnings and timings.")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class WordScore(BaseModel):
    word: str
    score: float = Field(..., description="Signed attribution toward the predicted class")
    importance: float = Field(..., description="0..1, relative to the strongest word")
    level: Literal["strong", "medium", "low", "none"]


class Explanation(BaseModel):
    method: str | None = Field(None, description="integrated_gradients or occlusion")
    reliable: bool | None = None
    baseline: str | None = None
    n_steps: int | None = None
    warning: str | None = None


class Timings(BaseModel):
    preprocessing_ms: float
    inference_ms: float
    explainability_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    prediction: Literal["HATE", "NON-HATE"]
    confidence: float
    language: str
    important_words: list[str]
    # only present when "details": true
    clean_text: str | None = None
    words: list[WordScore] | None = None
    explanation: Explanation | None = None
    warnings: list[str] | None = None
    timings_ms: Timings | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_loaded: bool
    device: str | None = None
    model_dir: str
    load_seconds: float | None = None
    error: str | None = None
