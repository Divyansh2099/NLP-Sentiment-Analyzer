"""
Utility functions for computing and formatting model evaluation metrics.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """Compute a full set of classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of label names (e.g. ['negative','neutral','positive']).

    Returns:
        Dictionary with accuracy, per-class precision/recall/f1, macro & weighted averages.

    Note:
        The ``labels`` parameter (integer indices) is passed to sklearn so that
        every class is always reported even when it is absent from *y_true* or
        *y_pred* (e.g. a model that never predicts "neutral").
    """
    if not y_true and not y_pred:
        # Empty inputs — return a zeroed-out report structure
        report: dict = {"accuracy": 0.0}
        for i, name in enumerate(labels):
            report[name] = {
                "precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0,
            }
        for avg in ("macro avg", "weighted avg"):
            report[avg] = {
                "precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0,
            }
        return report

    label_indices = list(range(len(labels)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report["accuracy"] = float(accuracy_score(y_true, y_pred))
    return report


def get_confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
    """Return a confusion matrix as a NumPy array."""
    return confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))


def save_metrics(metrics: dict, path: Path) -> None:
    """Save metrics dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def format_metrics_table(metrics: dict, labels: list) -> str:
    """Format metrics into a human-readable table string."""
    lines = []
    header = f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for label in labels:
        row = metrics.get(label, {})
        lines.append(
            f"{label:<12} {row.get('precision', 0):>10.4f} {row.get('recall', 0):>10.4f}"
            f" {row.get('f1-score', 0):>10.4f} {int(row.get('support', 0)):>10}"
        )

    lines.append("-" * len(header))
    lines.append(f"{'Accuracy':<12} {metrics.get('accuracy', 0):>10.4f}")

    for avg in ["macro avg", "weighted avg"]:
        row = metrics.get(avg, {})
        lines.append(
            f"{avg:<12} {row.get('precision', 0):>10.4f} {row.get('recall', 0):>10.4f}"
            f" {row.get('f1-score', 0):>10.4f} {int(row.get('support', 0)):>10}"
        )

    return "\n".join(lines)
