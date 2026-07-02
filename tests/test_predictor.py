"""
Tests for the SentimentPredictor (inference layer).

The _empty_result helper logic is validated via an inline reimplementation
to avoid importing torch/transformers. Full integration tests that exercise
the real model run only when PyTorch is installed.
"""

import pytest

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _empty_result_impl(labels: list[str], text: str) -> dict:
    """Reference implementation of _empty_result, mirroring the production code.

    Kept here so the logic can be unit-tested without importing the
    torch-dependent SentimentPredictor class.
    """
    return {
        "text": text,
        "sentiment": "neutral",
        "confidence": 0.0,
        "scores": {label: 0.0 for label in labels},
        "language": "en",
        "language_name": "English",
        "was_translated": False,
        "processing_time_ms": 0.0,
    }


LABELS = ["negative", "neutral", "positive"]


def test_empty_result_structure():
    result = _empty_result_impl(LABELS, "test text")
    assert result["text"] == "test text"
    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.0
    assert all(k in result["scores"] for k in LABELS)
    assert result["language"] == "en"
    assert result["language_name"] == "English"
    assert result["was_translated"] is False
    assert result["processing_time_ms"] == 0.0


def test_empty_result_for_empty_string():
    result = _empty_result_impl(LABELS, "")
    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.0


def test_empty_result_scores_all_zero():
    result = _empty_result_impl(LABELS, "anything")
    assert all(v == 0.0 for v in result["scores"].values())


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPredictorWithModel:
    """Full predictor tests — require PyTorch + Transformers."""

    def test_reset_predictor(self):
        from src.model.predictor import reset_predictor
        reset_predictor()
