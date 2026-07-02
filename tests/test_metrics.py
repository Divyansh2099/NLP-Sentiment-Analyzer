"""
Tests for the model evaluation metrics utilities.

These tests run without torch/transformers — they only exercise the
pure-Python metric formatting and computation helpers in src.utils.metrics.
"""

import json
from pathlib import Path

import pytest

from src.utils.metrics import (
    compute_metrics,
    format_metrics_table,
    get_confusion_matrix,
    save_metrics,
)

LABELS = ["negative", "neutral", "positive"]


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        metrics = compute_metrics(y_true, y_pred, LABELS)
        assert metrics["accuracy"] == 1.0
        for label in LABELS:
            assert metrics[label]["precision"] == 1.0
            assert metrics[label]["recall"] == 1.0
            assert metrics[label]["f1-score"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 0]
        y_pred = [2, 2, 2]
        metrics = compute_metrics(y_true, y_pred, LABELS)
        assert metrics["accuracy"] == 0.0

    def test_partial_correct(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2]  # 5/6 correct
        metrics = compute_metrics(y_true, y_pred, LABELS)
        assert metrics["accuracy"] == pytest.approx(5 / 6)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_includes_macro_and_weighted_avg(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        metrics = compute_metrics(y_true, y_pred, LABELS)
        assert "macro avg" in metrics
        assert "weighted avg" in metrics

    def test_empty_predictions_zero_division(self):
        """Empty predictions should not raise (zero_division=0)."""
        metrics = compute_metrics([], [], LABELS)
        assert metrics["accuracy"] == 0.0


class TestGetConfusionMatrix:
    """Tests for get_confusion_matrix."""

    def test_shape(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        cm = get_confusion_matrix(y_true, y_pred, LABELS)
        assert cm.shape == (3, 3)

    def test_diagonal_for_perfect(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        cm = get_confusion_matrix(y_true, y_pred, LABELS)
        # Diagonal should hold all counts, off-diagonal should be 0
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert cm[i][j] == 2
                else:
                    assert cm[i][j] == 0

    def test_off_diagonal_for_misclassification(self):
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]  # All wrong
        cm = get_confusion_matrix(y_true, y_pred, LABELS)
        assert cm[0][1] == 1  # True 0 predicted as 1
        assert cm[1][2] == 1  # True 1 predicted as 2
        assert cm[2][0] == 1  # True 2 predicted as 0


class TestSaveMetrics:
    """Tests for save_metrics."""

    def test_saves_valid_json(self, tmp_path):
        metrics = {"accuracy": 0.95, "negative": {"precision": 0.9}}
        path = tmp_path / "metrics.json"
        save_metrics(metrics, path)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == metrics

    def test_creates_parent_directory(self, tmp_path):
        metrics = {"accuracy": 0.5}
        path = tmp_path / "nested" / "dir" / "metrics.json"
        save_metrics(metrics, path)
        assert path.exists()


class TestFormatMetricsTable:
    """Tests for format_metrics_table."""

    def test_returns_string(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        metrics = compute_metrics(y_true, y_pred, LABELS)
        table = format_metrics_table(metrics, LABELS)
        assert isinstance(table, str)

    def test_table_includes_all_labels(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        metrics = compute_metrics(y_true, y_pred, LABELS)
        table = format_metrics_table(metrics, LABELS)
        for label in LABELS:
            assert label in table
        assert "Accuracy" in table
        assert "macro avg" in table

    def test_table_has_header_and_separator(self):
        metrics = compute_metrics([0, 1, 2], [0, 1, 2], LABELS)
        table = format_metrics_table(metrics, LABELS)
        lines = table.split("\n")
        assert "Precision" in lines[0]
        assert "-" * 10 in lines[1]
