#!/usr/bin/env python
"""
Model evaluation script — loads a saved model and evaluates on test data.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-path models/sentiment_classifier/best_model_epoch_3
    python scripts/evaluate.py --data-path data/processed/test_split.csv
"""

import argparse

import pandas as pd

from src.model.classifier import SentimentClassifier
from src.model.tokenizer import SentimentTokenizer
from src.model.evaluator import ModelEvaluator
from src.utils.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
)
from src.utils.logger import setup_logger

logger = setup_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Sentiment Classifier")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "sentiment_classifier"),
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PROCESSED_DIR / "test_split.csv"),
        help="Path to test data CSV",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Loading model...")
    model = SentimentClassifier.from_pretrained(args.model_path)
    tokenizer = SentimentTokenizer.from_pretrained(args.model_path)

    logger.info("Loading test data...")
    df = pd.read_csv(args.data_path)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    texts = df[text_col].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    evaluator = ModelEvaluator(model, tokenizer, device=args.device)
    results = evaluator.evaluate(texts, labels, batch_size=args.batch_size)

    print(f"\nTest Accuracy: {results['metrics']['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
