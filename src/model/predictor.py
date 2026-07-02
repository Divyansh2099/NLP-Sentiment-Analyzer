"""
Sentiment Predictor — inference wrapper that loads a trained model once
and provides single + batch prediction methods.

Handles language detection, translation, and edge cases transparently.
"""

import time
from pathlib import Path

import torch

from src.data.language_handler import process_multilingual, get_language_name
from src.model.classifier import SentimentClassifier
from src.model.tokenizer import SentimentTokenizer
from src.utils.config import (
    LABELS,
    MODEL_NAME,
    MODELS_DIR,
    NUM_LABELS,
)
from src.utils.logger import setup_logger

logger = setup_logger("model.predictor")

# ── Singleton Predictor ────────────────────────────────
_predictor_instance = None


class SentimentPredictor:
    """High-level inference wrapper for the sentiment classifier.

    Loads model and tokenizer on init, then provides:
        - predict(text) → single prediction dict
        - predict_batch(texts) → list of prediction dicts

    Thread-safe for use in FastAPI (no mutable shared state).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ):
        self.device = self._resolve_device(device)
        self.labels = LABELS[:NUM_LABELS]
        self.model_name = model_name

        # Load model + tokenizer
        if model_path is None:
            model_path = MODELS_DIR / "sentiment_classifier"

        model_path = Path(model_path)
        if model_path.exists():
            logger.info(f"Loading saved model from {model_path}")
            self.tokenizer = SentimentTokenizer.from_pretrained(model_path)
            self.model = SentimentClassifier.from_pretrained(model_path)
        else:
            logger.info(f"No saved model at {model_path} — loading base model from HuggingFace")
            self.tokenizer = SentimentTokenizer(model_name=model_name)
            self.model = SentimentClassifier(model_name=model_name, num_labels=NUM_LABELS)

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("SentimentPredictor ready for inference")

    def _resolve_device(self, device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict(self, text: str) -> dict:
        """Analyze sentiment of a single text.

        Automatically handles:
            - Language detection
            - Translation to English if needed
            - Empty/invalid input
            - Very long text (truncation)

        Args:
            text: Input text in any supported language.

        Returns:
            Dictionary with keys:
                - text: original input text
                - sentiment: label string ("positive", "neutral", "negative")
                - confidence: confidence score (0-1) for predicted class
                - scores: dict of per-class probabilities
                - language: detected language code
                - language_name: full language name
                - was_translated: whether text was translated
                - processing_time_ms: inference time in milliseconds
        """
        if not text or not text.strip():
            return self._empty_result(text or "")

        start_time = time.time()

        # Multilingual processing
        processed_text, detected_lang, was_translated = process_multilingual(text)

        # Tokenize
        encoding = self.tokenizer.encode_text(processed_text)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = self.model.get_probabilities(logits)
            pred_idx = self.model.predict_class(logits)

        processing_ms = (time.time() - start_time) * 1000

        # Build result
        probs_list = probs.squeeze(0).cpu().tolist()
        label_idx = pred_idx.item()

        result = {
            "text": text,
            "sentiment": self.labels[label_idx],
            "confidence": probs_list[label_idx],
            "scores": {
                self.labels[i]: round(probs_list[i], 4)
                for i in range(len(self.labels))
            },
            "language": detected_lang,
            "language_name": get_language_name(detected_lang),
            "was_translated": was_translated,
            "processing_time_ms": round(processing_ms, 2),
        }

        return result

    def predict_batch(self, texts: list[str]) -> dict:
        """Analyze sentiment of multiple texts.

        Args:
            texts: List of input text strings.

        Returns:
            Dictionary with keys:
                - results: list of per-text prediction dicts
                - total_time_ms: total processing time
                - count: number of texts processed
        """
        if not texts:
            return {"results": [], "total_time_ms": 0, "count": 0}

        start_time = time.time()

        # Process each text (language detection + translation)
        processed_texts = []
        langs = []
        translated_flags = []

        for text in texts:
            proc_text, lang, was_trans = process_multilingual(text)
            processed_texts.append(proc_text)
            langs.append(lang)
            translated_flags.append(was_trans)

        # Batch tokenize
        encoding = self.tokenizer.encode_batch(processed_texts)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Batch inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = self.model.get_probabilities(outputs["logits"])
            pred_indices = self.model.predict_class(outputs["logits"])

        total_ms = (time.time() - start_time) * 1000

        # Build results
        probs_np = probs.cpu().tolist()
        pred_np = pred_indices.cpu().tolist()

        results = []
        for i, text in enumerate(texts):
            label_idx = pred_np[i]
            results.append({
                "text": text,
                "sentiment": self.labels[label_idx],
                "confidence": round(probs_np[i][label_idx], 4),
                "scores": {
                    self.labels[j]: round(probs_np[i][j], 4)
                    for j in range(len(self.labels))
                },
                "language": langs[i],
                "language_name": get_language_name(langs[i]),
                "was_translated": translated_flags[i],
                "processing_time_ms": round(total_ms / len(texts), 2),
            })

        return {
            "results": results,
            "total_time_ms": round(total_ms, 2),
            "count": len(texts),
        }

    def _empty_result(self, text: str) -> dict:
        """Return a neutral result for empty/invalid input."""
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.0,
            "scores": {label: 0.0 for label in self.labels},
            "language": "en",
            "language_name": "English",
            "was_translated": False,
            "processing_time_ms": 0.0,
        }


def get_predictor(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> SentimentPredictor:
    """Get or create the global singleton predictor instance.

    Thread-safe: creates the predictor on first call and reuses it.

    Args:
        model_path: Path to saved model.
        device: Force device (cuda/cpu).

    Returns:
        SentimentPredictor instance.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor(
            model_path=model_path,
            device=device,
        )
    return _predictor_instance


def reset_predictor() -> None:
    """Reset the singleton predictor (for testing or model reload)."""
    global _predictor_instance
    _predictor_instance = None
