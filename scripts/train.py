#!/usr/bin/env python
"""
One-command training script for the BERT Sentiment Classifier.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 5 --batch-size 64 --lr 3e-5
    python scripts/train.py --max-length 128 --output-dir models/custom_model
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.model.classifier import SentimentClassifier
from src.model.tokenizer import SentimentTokenizer
from src.model.trainer import Trainer
from src.model.evaluator import ModelEvaluator
from src.utils.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    MAX_SEQ_LENGTH,
    LABELS,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MODEL_NAME,
    NUM_LABELS,
    REPORTS_DIR,
)
from src.utils.logger import setup_logger

logger = setup_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BERT Sentiment Classifier")

    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PROCESSED_DIR / "train_split.csv"),
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=str(DATA_PROCESSED_DIR / "val_split.csv"),
        help="Path to validation data CSV",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(DATA_PROCESSED_DIR / "test_split.csv"),
        help="Path to test data CSV (for final evaluation)",
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--num-labels", type=int, default=NUM_LABELS)
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "sentiment_classifier"))
    parser.add_argument("--no-eval", action="store_true", help="Skip final evaluation")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")

    return parser.parse_args()


def load_data(path: str) -> tuple[list[str], list[int]]:
    """Load a CSV dataset and return texts + labels."""
    df = pd.read_csv(path)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    texts = df[text_col].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    logger.info(f"Loaded {len(texts):,} samples from {path}")
    return texts, labels


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("NLP Sentiment Analyzer — Training Pipeline")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Max seq length: {args.max_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Output: {args.output_dir}")

    # Load data
    train_texts, train_labels = load_data(args.data_path)
    val_texts, val_labels = load_data(args.val_path)

    # Initialize tokenizer & model
    tokenizer = SentimentTokenizer(model_name=args.model_name, max_length=args.max_length)
    model = SentimentClassifier(
        model_name=args.model_name, num_labels=args.num_labels
    )
    logger.info(f"Parameters: {model.count_parameters()['trainable']:,} trainable")

    # Train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        device=args.device,
    )

    history = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )

    # Save training history
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Final evaluation on test set
    if not args.no_eval:
        logger.info("\n--- Final Evaluation on Test Set ---")
        test_texts, test_labels = load_data(args.test_path)

        # Reload best model — first try the explicit checkpoint, then the main dir
        best_model_path = Path(args.output_dir)
        best_subdirs = sorted(
            [d for d in best_model_path.iterdir() if d.is_dir() and d.name.startswith("best_model")],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if best_subdirs:
            best_model = SentimentClassifier.from_pretrained(best_subdirs[0])
            best_tokenizer = SentimentTokenizer.from_pretrained(best_subdirs[0])
            logger.info(f"Loaded best model from {best_subdirs[0]}")
        else:
            best_model = model
            best_tokenizer = tokenizer
            logger.info("No best_model checkpoint found — using final model")

        evaluator = ModelEvaluator(best_model, best_tokenizer, device=args.device)
        results = evaluator.evaluate(test_texts, test_labels, save_reports=True)

        logger.info(f"\n🏆 Test Accuracy: {results['metrics']['accuracy']*100:.2f}%")

    # Always save a final model to the main output directory for easy loading
    trainer.save_best_model()
    logger.info(f"Final model saved to {args.output_dir}")

    logger.info("\nTraining pipeline complete! ✅")


if __name__ == "__main__":
    main()
