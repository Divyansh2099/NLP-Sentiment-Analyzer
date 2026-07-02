"""
Integration tests for the FastAPI API endpoints.

These tests stub out the heavy model stack (torch/transformers) entirely,
so they run fast without requiring GPU or model weights. The predictor
is replaced with a mock that returns predictable results.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# ── Stub out heavy modules BEFORE importing the API ──────────────────────
# This lets us test the API layer in isolation (fast, no torch needed).
_mock_predictor = MagicMock()
_mock_predictor.predict.return_value = {
    "text": "This is great!",
    "sentiment": "positive",
    "confidence": 0.95,
    "scores": {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
    "language": "en",
    "language_name": "English",
    "was_translated": False,
    "processing_time_ms": 45.2,
}
_mock_predictor.predict_batch.return_value = {
    "results": [
        {
            "text": "Great!",
            "sentiment": "positive",
            "confidence": 0.90,
            "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.90},
            "language": "en",
            "language_name": "English",
            "was_translated": False,
            "processing_time_ms": 30.0,
        },
        {
            "text": "Terrible!",
            "sentiment": "negative",
            "confidence": 0.88,
            "scores": {"negative": 0.88, "neutral": 0.07, "positive": 0.05},
            "language": "en",
            "language_name": "English",
            "was_translated": False,
            "processing_time_ms": 30.0,
        },
    ],
    "total_time_ms": 60.0,
    "count": 2,
}
_mock_predictor.model = MagicMock()


def _install_predictor_stub():
    """Create a fake src.model.predictor module returning our mock."""
    fake_module = types.ModuleType("src.model.predictor")
    fake_module.SentimentPredictor = MagicMock()
    fake_module.get_predictor = lambda *a, **kw: _mock_predictor
    fake_module.reset_predictor = lambda *a, **kw: None
    sys.modules["src.model.predictor"] = fake_module


# Try to import torch; if missing, we must stub before importing api.main
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_install_predictor_stub()

from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Create a TestClient. The predictor is already stubbed at import time."""
    _mock_predictor.predict.reset_mock()
    _mock_predictor.predict_batch.reset_mock()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "languages_supported" in data
        assert "version" in data

    def test_health_lists_5_languages(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert len(data["languages_supported"]) == 5


class TestSupportedLanguagesEndpoint:
    def test_returns_200(self, client):
        response = client.get("/api/v1/supported-languages")
        assert response.status_code == 200

    def test_returns_correct_count(self, client):
        response = client.get("/api/v1/supported-languages")
        data = response.json()
        assert data["count"] == 5
        assert len(data["languages"]) == 5


class TestPredictEndpoint:
    def test_predict_single_text(self, client):
        response = client.post("/api/v1/predict", json={"text": "This is great!"})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.95
        assert "scores" in data

    def test_predict_response_structure(self, client):
        response = client.post("/api/v1/predict", json={"text": "Hello"})
        data = response.json()
        assert "text" in data
        assert "sentiment" in data
        assert "confidence" in data
        assert "scores" in data
        assert "language" in data
        assert "language_name" in data
        assert "was_translated" in data
        assert "processing_time_ms" in data

    def test_predict_empty_text_rejected(self, client):
        response = client.post("/api/v1/predict", json={"text": ""})
        assert response.status_code == 422  # Validation error

    def test_predict_missing_text_field(self, client):
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_predict_oversized_text_rejected(self, client):
        response = client.post(
            "/api/v1/predict",
            json={"text": "a" * 5001},
        )
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_multiple(self, client):
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["Great!", "Terrible!"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_predict_empty_list_rejected(self, client):
        response = client.post("/api/v1/predict/batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_predict_too_many_rejected(self, client):
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["text"] * 101},
        )
        assert response.status_code == 422

    def test_batch_predict_response_structure(self, client):
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["test"]},
        )
        data = response.json()
        assert "results" in data
        assert "total_time_ms" in data
        assert "count" in data


class TestApiDocs:
    def test_swagger_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/predict" in schema["paths"]
