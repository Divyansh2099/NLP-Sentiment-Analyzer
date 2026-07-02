"""
Tests for the BERT classifier module.

NOTE: These tests require PyTorch + Transformers to be installed.
They are marked as 'slow' since they load the model architecture.
Run with: pytest tests/test_classifier.py -m "not slow" to skip.
"""

import pytest
from unittest.mock import patch, MagicMock

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping classifier tests")


class TestSentimentClassifier:
    """Tests for the SentimentClassifier class."""

    @patch("src.model.classifier.BertForSequenceClassification.from_pretrained")
    def test_init_loads_bert(self, mock_bert):
        mock_bert.return_value = MagicMock()

        from src.model.classifier import SentimentClassifier

        classifier = SentimentClassifier(num_labels=3)
        assert classifier.num_labels == 3
        assert classifier.labels == ["negative", "neutral", "positive"]
        mock_bert.assert_called_once()

    @patch("src.model.classifier.BertForSequenceClassification.from_pretrained")
    def test_get_probabilities(self, mock_bert):
        mock_bert.return_value = MagicMock()

        from src.model.classifier import SentimentClassifier

        classifier = SentimentClassifier(num_labels=3)
        logits = torch.tensor([[2.0, 0.5, -1.0]])
        probs = classifier.get_probabilities(logits)

        assert probs.shape == (1, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))

    @patch("src.model.classifier.BertForSequenceClassification.from_pretrained")
    def test_predict_class(self, mock_bert):
        mock_bert.return_value = MagicMock()

        from src.model.classifier import SentimentClassifier

        classifier = SentimentClassifier(num_labels=3)
        logits = torch.tensor([[2.0, 0.5, -1.0]])
        pred = classifier.predict_class(logits)

        assert pred.item() == 0  # Highest logit is at index 0

    @patch("src.model.classifier.BertForSequenceClassification.from_pretrained")
    def test_count_parameters(self, mock_bert):
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        mock_bert.return_value = mock_model

        from src.model.classifier import SentimentClassifier

        classifier = SentimentClassifier(num_labels=3)
        params = classifier.count_parameters()

        assert "total" in params
        assert "trainable" in params
        assert params["trainable"] > 0
