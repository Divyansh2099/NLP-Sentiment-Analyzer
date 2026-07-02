#!/usr/bin/env python
"""
One-command script to generate the full synthetic dataset.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n-per-class 50000
    python scripts/generate_data.py --n-per-class 50000 --multilingual 1000

This creates balanced, preprocessed train/val/test splits in data/processed/.
No external downloads or API keys required.
"""

import argparse
import time

from src.data.dataset_builder import generate_and_build
from src.utils.logger import setup_logger

logger = setup_logger("generate_data")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sentiment analysis dataset"
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=30_000,
        help="English reviews per sentiment class (default: 30000)",
    )
    parser.add_argument(
        "--multilingual",
        type=int,
        default=500,
        help="Multilingual samples per language per class (default: 500)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.2,
        help="Probability of noise injection (emojis, typos, etc.) (default: 0.2)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  NLP Sentiment Analyzer — Dataset Generator")
    print("=" * 60)
    print(f"  English reviews per class: {args.n_per_class:,}")
    print(f"  Multilingual samples:      {args.multilingual:,} × 4 langs × 3 classes")
    print(f"  Noise level:               {args.noise_level:.0%}")
    print("=" * 60)
    print()

    start = time.time()

    train, val, test = generate_and_build(
        n_english_per_class=args.n_per_class,
        n_multilingual_per_class=args.multilingual,
        noise_level=args.noise_level,
    )

    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"  ✅ Dataset generated in {elapsed:.1f}s")
    print(f"     Train:      {len(train):>10,} samples")
    print(f"     Validation: {len(val):>10,} samples")
    print(f"     Test:       {len(test):>10,} samples")
    print(f"     Total:      {len(train) + len(val) + len(test):>10,} samples")
    print()
    print("  Output files:")
    print("    data/processed/combined_dataset.csv")
    print("    data/processed/train_split.csv")
    print("    data/processed/val_split.csv")
    print("    data/processed/test_split.csv")
    print()
    print("  Next step: python scripts/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
