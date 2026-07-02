"""
Dataset downloader — fetches raw datasets from HuggingFace and Kaggle sources.
Supports Amazon Reviews, Twitter Sentiment140, and Hotel Reviews.
"""

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.config import DATA_RAW_DIR
from src.utils.logger import setup_logger

logger = setup_logger("data.downloader")


def download_amazon_reviews(max_samples: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Download Amazon product reviews from HuggingFace `amazon_polarity` dataset.

    The dataset has 2 classes (positive=1, negative=0). We map:
      0 (negative) → 0 (negative)
      1 (positive) → 2 (positive)

    Args:
        max_samples: Maximum number of samples to download (balanced).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: text, label, source
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install 'datasets' package: pip install datasets")
        raise

    logger.info("Downloading Amazon Reviews dataset from HuggingFace...")
    dataset = load_dataset("amazon_polarity", split="train[:5%]")  # ~140K subset

    df = pd.DataFrame({
        "text": dataset["content"],
        "label": dataset["label"],
    })

    # Map Amazon labels to our 3-class scheme (0=negative, 2=positive)
    df["label"] = df["label"].map({0: 0, 1: 2})

    # Balance and sample
    half = max_samples // 2
    df_neg = df[df["label"] == 0].sample(n=half, random_state=seed)
    df_pos = df[df["label"] == 2].sample(n=half, random_state=seed)
    df = pd.concat([df_neg, df_pos], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    df["source"] = "amazon"
    logger.info(f"Amazon Reviews: {len(df)} samples loaded")
    return df


def download_twitter_sentiment(max_samples: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Download Sentiment140 Twitter dataset from HuggingFace.

    Labels: 0=negative, 2=positive, 4=positive (mapped to our scheme).

    Args:
        max_samples: Maximum number of samples to download.
        seed: Random seed.

    Returns:
        DataFrame with columns: text, label, source
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install 'datasets' package: pip install datasets")
        raise

    logger.info("Downloading Twitter Sentiment140 dataset from HuggingFace...")
    dataset = load_dataset("sentiment140", split="train")

    df = pd.DataFrame({
        "text": dataset["text"],
        "label": dataset["label"],
    })

    # Map: 0→0 (negative), 2→1 (neutral), 4→2 (positive)
    label_map = {0: 0, 2: 1, 4: 2}
    df = df[df["label"].isin(label_map.keys())]
    df["label"] = df["label"].map(label_map)

    # Balance across classes
    per_class = max_samples // 3
    dfs = []
    for lbl in [0, 1, 2]:
        subset = df[df["label"] == lbl]
        n = min(per_class, len(subset))
        dfs.append(subset.sample(n=n, random_state=seed))
    df = pd.concat(dfs, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    df["source"] = "twitter"
    logger.info(f"Twitter Sentiment: {len(df)} samples loaded")
    return df


def download_hotel_reviews(max_samples: int = 30_000, seed: int = 42) -> pd.DataFrame:
    """Download hotel/restaurant reviews with 3-class sentiment.

    Uses the `carblacac/twitter-sentiment-analysis` dataset (self-annotated
    tweets & reviews with 3 classes) as a proxy for hotel reviews, or generates
    synthetic hotel-style reviews based on templates.

    Args:
        max_samples: Maximum number of samples.
        seed: Random seed.

    Returns:
        DataFrame with columns: text, label, source
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install 'datasets' package: pip install datasets")
        raise

    logger.info("Downloading Hotel Reviews dataset from HuggingFace...")
    dataset = load_dataset("carblacac/twitter-sentiment-analysis",
                           trust_remote_code=True, split="train")

    df = pd.DataFrame({
        "text": dataset["text"],
        "label": dataset["feeling"],
    })

    # Map labels: 0=negative, 1=neutral, 2=positive
    df = df[df["label"].isin([0, 1, 2])]
    df["label"] = df["label"].astype(int)

    # Balance and sample
    per_class = max_samples // 3
    dfs = []
    for lbl in [0, 1, 2]:
        subset = df[df["label"] == lbl]
        n = min(per_class, len(subset))
        if n > 0:
            dfs.append(subset.sample(n=n, random_state=seed))
    df = pd.concat(dfs, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    df["source"] = "hotel"
    logger.info(f"Hotel Reviews: {len(df)} samples loaded")
    return df


def download_all_datasets(
    amazon_samples: int = 100_000,
    twitter_samples: int = 50_000,
    hotel_samples: int = 30_000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Download all datasets and return as a dictionary.

    Returns:
        Dictionary mapping source name to DataFrame.
    """
    datasets = {}

    datasets["amazon"] = download_amazon_reviews(
        max_samples=amazon_samples, seed=seed
    )
    datasets["twitter"] = download_twitter_sentiment(
        max_samples=twitter_samples, seed=seed
    )
    datasets["hotel"] = download_hotel_reviews(
        max_samples=hotel_samples, seed=seed
    )

    total = sum(len(df) for df in datasets.values())
    logger.info(f"Total raw samples downloaded: {total:,}")
    for name, df in datasets.items():
        logger.info(f"  {name}: {len(df):,} samples")
        label_counts = df["label"].value_counts().sort_index()
        for lbl, count in label_counts.items():
            logger.info(f"    label {lbl}: {count:,}")

    return datasets


if __name__ == "__main__":
    raw_datasets = download_all_datasets()
    for name, df in raw_datasets.items():
        out_path = DATA_RAW_DIR / f"{name}_reviews.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {name} to {out_path}")
