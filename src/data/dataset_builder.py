"""
Dataset builder — combines multiple preprocessed datasets into a
single balanced, stratified dataset ready for model training.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.preprocessor import preprocess_dataframe
from src.utils.config import DATA_PROCESSED_DIR
from src.utils.logger import setup_logger

logger = setup_logger("data.dataset_builder")

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def combine_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple DataFrames into one unified dataset.

    Args:
        datasets: Dictionary mapping source name to DataFrame.
                  Each DataFrame must have 'clean_text', 'label', 'source'.

    Returns:
        Combined DataFrame.
    """
    frames = []
    for name, df in datasets.items():
        if "clean_text" not in df.columns:
            logger.warning(f"Dataset '{name}' missing 'clean_text' — skipping.")
            continue
        if "source" not in df.columns:
            df["source"] = name
        frames.append(df[["clean_text", "label", "source"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Combined dataset: {len(combined):,} samples from {len(frames)} sources")
    return combined


def balance_classes(
    df: pd.DataFrame,
    label_column: str = "label",
    target_per_class: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Balance class distribution by undersampling majority classes.

    Args:
        df: DataFrame with label column.
        label_column: Name of the label column.
        target_per_class: Max samples per class. If None, uses the minimum class count.
        random_state: Random seed.

    Returns:
        Balanced DataFrame.
    """
    class_counts = df[label_column].value_counts().sort_index()
    logger.info(f"Class distribution before balancing:")
    for lbl, count in class_counts.items():
        name = LABEL_NAMES.get(lbl, str(lbl))
        logger.info(f"  {name} ({lbl}): {count:,}")

    if target_per_class is None:
        target_per_class = class_counts.min()

    dfs = []
    for lbl in class_counts.index:
        subset = df[df[label_column] == lbl]
        n = min(target_per_class, len(subset))
        dfs.append(subset.sample(n=n, random_state=random_state))

    balanced = pd.concat(dfs, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Class distribution after balancing (target={target_per_class:,}):")
    for lbl, count in balanced[label_column].value_counts().sort_index().items():
        name = LABEL_NAMES.get(lbl, str(lbl))
        logger.info(f"  {name} ({lbl}): {count:,}")

    return balanced


def split_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets (stratified).

    Args:
        df: Full dataset.
        label_column: Name of the label column.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining after test split).
        random_state: Random seed.

    Returns:
        (train_df, val_df, test_df)
    """
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state,
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[label_column],
        random_state=random_state,
    )

    logger.info(f"Dataset splits:")
    logger.info(f"  Train:      {len(train):,} ({len(train) / len(df) * 100:.1f}%)")
    logger.info(f"  Validation: {len(val):,} ({len(val) / len(df) * 100:.1f}%)")
    logger.info(f"  Test:       {len(test):,} ({len(test) / len(df) * 100:.1f}%)")

    return train, val, test


def build_dataset(
    datasets: dict[str, pd.DataFrame],
    target_per_class: int | None = None,
    test_size: float = 0.10,
    val_size: float = 0.10,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full dataset building pipeline: combine → balance → split → save.

    Args:
        datasets: Dictionary of source DataFrames.
        target_per_class: Max samples per class after balancing.
        test_size: Test set fraction.
        val_size: Validation set fraction.
        output_path: Path to save the full combined CSV (before split).

    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("Starting dataset build pipeline...")

    # Combine
    combined = combine_datasets(datasets)

    # Balance
    balanced = balance_classes(combined, target_per_class=target_per_class)

    # Split
    train, val, test = split_dataset(
        balanced, test_size=test_size, val_size=val_size
    )

    # Save processed dataset
    if output_path is None:
        output_path = DATA_PROCESSED_DIR / "combined_dataset.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full = pd.concat([train, val, test], ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)
    full.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved combined dataset to {output_path}")

    # Save splits
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_path = output_path.parent / f"{name}_split.csv"
        split_df.to_csv(split_path, index=False, encoding="utf-8")
        logger.info(f"Saved {name} split to {split_path}")

    return train, val, test


def generate_and_build(
    n_english_per_class: int = 30_000,
    n_multilingual_per_class: int = 500,
    noise_level: float = 0.2,
    target_per_class: int | None = None,
    test_size: float = 0.10,
    val_size: float = 0.10,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-command pipeline: generate synthetic reviews → preprocess → balance → split.

    No external downloads required. Uses the built-in generator to create a
    self-contained dataset of customer reviews across 6 domains and 5 languages.

    Args:
        n_english_per_class: English reviews per sentiment class.
        n_multilingual_per_class: Multilingual samples per language per class.
        noise_level: Probability of noise injection in generated text.
        target_per_class: Max samples per class after balancing.
        test_size: Test set fraction.
        val_size: Validation set fraction.
        output_path: Path to save the full combined CSV.

    Returns:
        (train_df, val_df, test_df)
    """
    from src.data.generator import generate_full_dataset

    logger.info("=" * 60)
    logger.info("Generating self-contained synthetic dataset")
    logger.info("=" * 60)

    # Generate
    raw_df = generate_full_dataset(
        n_english_per_class=n_english_per_class,
        n_multilingual_per_class_per_lang=n_multilingual_per_class,
        noise_level=noise_level,
    )

    # Preprocess
    processed_df = preprocess_dataframe(raw_df, text_column="text")
    processed_df["source"] = "generated"

    # Build
    return build_dataset(
        {"generated": processed_df},
        target_per_class=target_per_class,
        test_size=test_size,
        val_size=val_size,
        output_path=output_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the sentiment dataset")
    parser.add_argument(
        "--source",
        choices=["generate", "download"],
        default="generate",
        help="'generate' for self-contained synthetic data, 'download' to fetch from HuggingFace",
    )
    parser.add_argument("--n-per-class", type=int, default=30_000,
                        help="Reviews per sentiment class (generation mode)")
    parser.add_argument("--multilingual", type=int, default=500,
                        help="Multilingual samples per language per class")
    args = parser.parse_args()

    if args.source == "generate":
        train, val, test = generate_and_build(
            n_english_per_class=args.n_per_class,
            n_multilingual_per_class=args.multilingual,
        )
    else:
        from src.data.downloader import download_all_datasets

        raw = download_all_datasets()
        processed = {}
        for name, df in raw.items():
            df = preprocess_dataframe(df, text_column="text")
            processed[name] = df
        train, val, test = build_dataset(processed)

    print(f"\nFinal: {len(train)} train, {len(val)} val, {len(test)} test")
