"""
Model evaluator — computes detailed metrics, generates confusion matrix
plots, and saves performance reports.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.classifier import SentimentClassifier
from src.model.tokenizer import SentimentTokenizer
from src.utils.config import LABELS, REPORTS_DIR
from src.utils.logger import setup_logger
from src.utils.metrics import (
    compute_metrics,
    get_confusion_matrix,
    save_metrics,
    format_metrics_table,
)

logger = setup_logger("model.evaluator")


class ModelEvaluator:
    """Evaluates the sentiment classifier on test data."""

    def __init__(
        self,
        model: SentimentClassifier,
        tokenizer: SentimentTokenizer,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        texts: list[str],
        labels: list[int],
        batch_size: int = 64,
        save_reports: bool = True,
    ) -> dict:
        """Run full evaluation pipeline.

        Args:
            texts: Test texts.
            labels: Ground-truth labels.
            batch_size: Batch size for inference.
            save_reports: Whether to save reports to disk.

        Returns:
            Dictionary with all metrics and metadata.
        """
        logger.info(f"Evaluating on {len(texts):,} samples...")

        # Run inference
        all_preds, all_labels, all_probs, times = self._predict_all(
            texts, labels, batch_size
        )

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, LABELS)
        conf_matrix = get_confusion_matrix(all_labels, all_preds, LABELS)

        result = {
            "metrics": metrics,
            "confusion_matrix": conf_matrix.tolist(),
            "num_samples": len(texts),
            "avg_inference_ms": np.mean(times),
            "p95_inference_ms": np.percentile(times, 95),
            "labels": LABELS,
        }

        # Print results
        logger.info("\n" + format_metrics_table(metrics, LABELS))
        logger.info(
            f"Avg inference time: {result['avg_inference_ms']:.1f}ms | "
            f"P95: {result['p95_inference_ms']:.1f}ms"
        )

        # Save reports
        if save_reports:
            self._save_reports(result)

        return result

    def _predict_all(
        self, texts: list[str], labels: list[int], batch_size: int
    ) -> tuple[list[int], list[int], list[list[float]], list[float]]:
        """Run batched inference and collect predictions.

        Returns:
            (predictions, ground_truth, probabilities, inference_times_ms)
        """
        all_preds = []
        all_labels = list(labels)
        all_probs = []
        times = []

        self.model.eval()
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
            batch_texts = texts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            # Tokenize
            encoded = self.tokenizer.encode_batch(batch_texts)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Inference
            start = time.time()
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]
                probs = self.model.get_probabilities(logits)
                preds = self.model.predict_class(logits)
            elapsed_ms = (time.time() - start) * 1000

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            times.append(elapsed_ms)

        return all_preds, all_labels, all_probs, times

    def _save_reports(self, result: dict) -> None:
        """Save metrics JSON, confusion matrix plot, and performance report."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Metrics JSON
        metrics_path = REPORTS_DIR / "training_metrics.json"
        save_metrics(result["metrics"], metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")

        # Full result JSON (with confusion matrix and timing)
        full_path = REPORTS_DIR / "evaluation_results.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Full results saved to {full_path}")

        # Confusion matrix plot
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns

            conf = np.array(result["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                conf,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=LABELS,
                yticklabels=LABELS,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix — Sentiment Classifier")
            plt.tight_layout()
            plot_path = REPORTS_DIR / "confusion_matrix.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Confusion matrix saved to {plot_path}")
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping confusion matrix plot")

        # Performance markdown report
        self._save_performance_report(result)

    def _save_performance_report(self, result: dict) -> None:
        """Save a human-readable markdown performance report."""
        report_path = REPORTS_DIR / "model_performance.md"
        metrics = result["metrics"]

        lines = [
            "# Sentiment Classifier — Performance Report\n",
            f"**Model**: BERT (`bert-base-multilingual-cased`)",
            f"**Test Samples**: {result['num_samples']:,}",
            f"**Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
            f"**Avg Inference Time**: {result['avg_inference_ms']:.1f}ms",
            f"**P95 Inference Time**: {result['p95_inference_ms']:.1f}ms\n",
            "## Per-Class Metrics\n",
            "| Class | Precision | Recall | F1-Score | Support |",
            "|-------|-----------|--------|----------|---------|",
        ]

        for label in LABELS:
            m = metrics.get(label, {})
            lines.append(
                f"| {label.capitalize()} | {m.get('precision', 0):.4f} | "
                f"{m.get('recall', 0):.4f} | {m.get('f1-score', 0):.4f} | "
                f"{int(m.get('support', 0))} |"
            )

        lines.append("\n## Confusion Matrix\n")
        lines.append("```")
        header = "        " + "  ".join(f"{l:>10}" for l in LABELS)
        lines.append(header)
        for i, label in enumerate(LABELS):
            row = f"{label:>8}" + "".join(f"{v:>10}" for v in result["confusion_matrix"][i])
            lines.append(row)
        lines.append("```\n")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Performance report saved to {report_path}")
