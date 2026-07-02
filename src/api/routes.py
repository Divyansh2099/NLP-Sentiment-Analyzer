"""
API route definitions for the Sentiment Analysis API.
"""

import time
from typing import Optional

from fastapi import APIRouter, Request

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    LanguageInfo,
    PredictRequest,
    PredictResponse,
    SentimentScores,
    SupportedLanguagesResponse,
)
from src.data.language_handler import get_supported_languages
from src.model.predictor import get_predictor

router = APIRouter(prefix="/api/v1")

API_VERSION = "1.0.0"


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Analyze sentiment of a single text",
    description="Classifies the sentiment of a single text input. "
    "Supports 5 languages: English, Spanish, French, German, Portuguese.",
    responses={400: {"description": "Invalid input"}, 503: {"description": "Model not loaded"}},
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Analyze sentiment of a single text."""
    predictor = get_predictor()
    result = predictor.predict(request.text)

    return PredictResponse(
        text=result["text"],
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        scores=SentimentScores(**result["scores"]),
        language=result["language"],
        language_name=result["language_name"],
        was_translated=result["was_translated"],
        processing_time_ms=result["processing_time_ms"],
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Analyze sentiment of multiple texts",
    description="Classifies sentiment of up to 100 texts at once. "
    "Returns individual results with per-class scores.",
    responses={400: {"description": "Invalid input"}, 503: {"description": "Model not loaded"}},
)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Analyze sentiment of multiple texts."""
    predictor = get_predictor()
    result = predictor.predict_batch(request.texts)

    results = [
        PredictResponse(
            text=r["text"],
            sentiment=r["sentiment"],
            confidence=r["confidence"],
            scores=SentimentScores(**r["scores"]),
            language=r["language"],
            language_name=r["language_name"],
            was_translated=r["was_translated"],
            processing_time_ms=r["processing_time_ms"],
        )
        for r in result["results"]
    ]

    return BatchPredictResponse(
        results=results,
        total_time_ms=result["total_time_ms"],
        count=result["count"],
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status, model readiness, and supported languages.",
)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    try:
        predictor = get_predictor()
        model_loaded = getattr(predictor, "_loaded", predictor.model is not None)
    except Exception:
        model_loaded = False

    languages = [
        LanguageInfo(code=lang["code"], name=lang["name"])
        for lang in get_supported_languages()
    ]

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        languages_supported=languages,
        version=API_VERSION,
    )


@router.get(
    "/supported-languages",
    response_model=SupportedLanguagesResponse,
    summary="List supported languages",
    description="Returns the list of languages supported for sentiment analysis.",
)
async def supported_languages() -> SupportedLanguagesResponse:
    """List supported languages for sentiment analysis."""
    languages = [
        LanguageInfo(code=lang["code"], name=lang["name"])
        for lang in get_supported_languages()
    ]

    return SupportedLanguagesResponse(
        languages=languages,
        count=len(languages),
    )
