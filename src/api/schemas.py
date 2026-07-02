"""
Pydantic request/response schemas for the Sentiment Analysis API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for single text sentiment prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment.",
        examples=["This product is absolutely wonderful! Highly recommend it."],
    )


class BatchPredictRequest(BaseModel):
    """Request body for batch sentiment prediction."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze (max 100).",
        examples=[
            [
                "Great service, loved it!",
                "Terrible experience, will not return.",
                "It was okay, nothing special.",
            ]
        ],
    )


class SentimentScores(BaseModel):
    """Per-class probability scores."""

    negative: float = Field(..., ge=0.0, le=1.0, description="Probability of negative sentiment")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Probability of neutral sentiment")
    positive: float = Field(..., ge=0.0, le=1.0, description="Probability of positive sentiment")


class PredictResponse(BaseModel):
    """Response for a single sentiment prediction."""

    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence for predicted class")
    scores: SentimentScores = Field(..., description="Per-class probabilities")
    language: str = Field(..., description="Detected language code (ISO 639-1)")
    language_name: str = Field(..., description="Full language name")
    was_translated: bool = Field(..., description="Whether text was translated from another language")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictResponse(BaseModel):
    """Response for batch sentiment prediction."""

    results: list[PredictResponse] = Field(..., description="Individual prediction results")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    count: int = Field(..., description="Number of texts processed")


class LanguageInfo(BaseModel):
    """Information about a supported language."""

    code: str = Field(..., description="ISO 639-1 language code")
    name: str = Field(..., description="Full language name")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    languages_supported: list[LanguageInfo] = Field(..., description="List of supported languages")
    version: str = Field(..., description="API version")


class SupportedLanguagesResponse(BaseModel):
    """Response listing supported languages."""

    languages: list[LanguageInfo] = Field(..., description="List of supported languages")
    count: int = Field(..., description="Number of supported languages")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error description")
    status_code: int = Field(..., description="HTTP status code")
