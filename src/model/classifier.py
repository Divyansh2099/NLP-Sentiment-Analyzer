"""
BERT-based sentiment classifier module.
Wraps HuggingFace's BertForSequenceClassification with custom save/load logic.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from src.utils.config import MODEL_NAME, NUM_LABELS, LABELS
from src.utils.logger import setup_logger

logger = setup_logger("model.classifier")


class SentimentClassifier(nn.Module):
    """BERT-based multi-class sentiment classifier.

    Uses `bert-base-multilingual-cased` as backbone with a classification
    head for 3 classes: negative (0), neutral (1), positive (2).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.labels = LABELS[:num_labels]

        # Load pre-trained BERT with classification head
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )

        # Add dropout if not already handled by the model
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"Classifier initialized: {model_name}, "
            f"labels={num_labels}, dropout={dropout}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass through BERT + classification head.

        Args:
            input_ids: Token IDs tensor (batch_size, seq_len).
            attention_mask: Attention mask tensor (batch_size, seq_len).
            labels: Optional ground-truth labels for loss computation.

        Returns:
            Dictionary with logits and (optionally) loss.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        result = {"logits": logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to softmax probabilities.

        Args:
            logits: Raw model output logits.

        Returns:
            Softmax probabilities tensor.
        """
        return torch.softmax(logits, dim=-1)

    def predict_class(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices from logits.

        Args:
            logits: Raw model output logits.

        Returns:
            Tensor of predicted class indices.
        """
        return torch.argmax(logits, dim=-1)

    def save(self, path: str | Path) -> None:
        """Save model weights to disk.

        Args:
            path: Directory path to save model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save the underlying BERT model
        self.bert.save_pretrained(path)
        # Save config
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "labels": self.labels,
        }
        torch.save(config, path / "classifier_config.pt")
        logger.info(f"Model saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str | Path):
        """Load a saved model from disk.

        Args:
            path: Directory path where model was saved.

        Returns:
            SentimentClassifier instance with loaded weights.
        """
        path = Path(path)
        # Load config
        config_path = path / "classifier_config.pt"
        if config_path.exists():
            config = torch.load(config_path, map_location="cpu")
            model_name = config.get("model_name", str(path))
            num_labels = config.get("num_labels", NUM_LABELS)
        else:
            model_name = str(path)
            num_labels = NUM_LABELS

        instance = cls(model_name=model_name, num_labels=num_labels)
        # Load weights into the BERT model
        instance.bert = BertForSequenceClassification.from_pretrained(str(path))
        logger.info(f"Model loaded from {path}")
        return instance

    def count_parameters(self) -> dict:
        """Count trainable and total parameters.

        Returns:
            Dictionary with parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
