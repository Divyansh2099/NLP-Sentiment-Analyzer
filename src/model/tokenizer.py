"""
Tokenizer wrapper around HuggingFace's pre-trained BERT tokenizer.
Handles encoding, padding, and truncation for sentiment classification.
"""

import os
from pathlib import Path

from transformers import AutoTokenizer

from src.utils.config import MODEL_NAME, MAX_SEQ_LENGTH
from src.utils.logger import setup_logger

logger = setup_logger("model.tokenizer")


class SentimentTokenizer:
    """Wrapper around AutoTokenizer for sentiment analysis tasks."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer()
        vocab_size = getattr(self.tokenizer, "vocab_size", "N/A")
        logger.info(
            f"Tokenizer loaded: {model_name} (max_length={max_length}, "
            f"vocab_size={vocab_size})"
        )

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer, downloading if necessary."""
        return AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )

    def encode_text(self, text: str) -> dict:
        """Encode a single text string.

        Args:
            text: Input text.

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def encode_batch(self, texts: list[str]) -> dict:
        """Encode a batch of text strings.

        Args:
            texts: List of input texts.

        Returns:
            Dictionary with input_ids, attention_mask tensors.
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def decode(self, input_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            input_ids: Token ID sequence.
            skip_special_tokens: Whether to skip [CLS], [SEP], etc.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        """Save tokenizer to disk.

        Args:
            path: Directory path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str | Path, max_length: int = MAX_SEQ_LENGTH):
        """Load a saved tokenizer from disk.

        Args:
            path: Directory path where tokenizer was saved.
            max_length: Override max sequence length.

        Returns:
            SentimentTokenizer instance.
        """
        instance = cls.__new__(cls)
        instance.model_name = str(path)
        instance.max_length = max_length
        instance.tokenizer = AutoTokenizer.from_pretrained(str(path))
        logger.info(f"Tokenizer loaded from {path}")
        return instance
