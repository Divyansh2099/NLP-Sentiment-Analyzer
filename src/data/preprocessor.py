"""
Text preprocessing pipeline for sentiment analysis.
Handles cleaning, normalization, and deduplication.
"""

import re
import html

import pandas as pd
from tqdm import tqdm

from src.utils.logger import setup_logger

logger = setup_logger("data.preprocessor")

# ── Regex Patterns ──────────────────────────────────────
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
SPECIAL_CHARS = re.compile(r"[^\w\s.,!?;:'\"()\-\[\]{}]")
MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply full cleaning pipeline to a single text string.

    Steps:
        1. Unescape HTML entities
        2. Remove HTML tags
        3. Remove URLs
        4. Remove @mentions and #hashtags (keep word portion)
        5. Replace emojis with text placeholder
        6. Remove excessive special characters
        7. Normalize whitespace
        8. Strip and lowercase

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unescape HTML
    text = html.unescape(text)

    # Remove HTML tags
    text = HTML_TAG_PATTERN.sub("", text)

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove mentions, keep hashtag words
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub(lambda m: m.group(0)[1:], text)

    # Basic emoji removal (Unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # Remove excessive special characters (keep basic punctuation)
    text = SPECIAL_CHARS.sub("", text)

    # Normalize whitespace
    text = MULTI_SPACE.sub(" ", text)

    # Strip and lowercase
    text = text.strip().lower()

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    min_text_length: int = 3,
    max_text_length: int = 512,
) -> pd.DataFrame:
    """Preprocess a full DataFrame: clean text, remove invalid rows, deduplicate.

    Args:
        df: Raw DataFrame.
        text_column: Name of the text column.
        label_column: Name of the label column.
        min_text_length: Drop texts shorter than this.
        max_text_length: Truncate texts longer than this.

    Returns:
        Cleaned DataFrame.
    """
    original_count = len(df)
    logger.info(f"Preprocessing {original_count:,} rows from source...")

    # Drop rows with missing values
    df = df.dropna(subset=[text_column, label_column]).copy()

    # Clean text
    tqdm.pandas(desc="Cleaning text")
    df["clean_text"] = df[text_column].progress_apply(clean_text)

    # Filter by length
    df["text_length"] = df["clean_text"].str.len()
    df = df[
        (df["text_length"] >= min_text_length)
        & (df["text_length"] <= max_text_length)
    ].copy()

    # Drop duplicates based on cleaned text
    df = df.drop_duplicates(subset=["clean_text", label_column]).copy()

    # Truncate long texts
    df["clean_text"] = df["clean_text"].str.slice(0, max_text_length)

    # Drop helper column
    df = df.drop(columns=["text_length"])

    final_count = len(df)
    removed = original_count - final_count
    logger.info(
        f"Preprocessing complete: {final_count:,} rows "
        f"(removed {removed:,}, {removed / max(original_count, 1) * 100:.1f}%)"
    )

    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    """Ensure labels are integers in the range [0, NUM_LABELS).

    Args:
        df: DataFrame with label column.
        label_column: Name of the label column.

    Returns:
        DataFrame with integer labels.

    Raises:
        ValueError: If any label falls outside the valid range [0, NUM_LABELS).
    """
    from src.utils.config import NUM_LABELS

    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df.dropna(subset=[label_column])
    df[label_column] = df[label_column].astype(int)

    invalid = df[~df[label_column].between(0, NUM_LABELS - 1)]
    if not invalid.empty:
        n_invalid = len(invalid)
        logger.warning(
            f"Dropping {n_invalid} rows with labels outside range [0, {NUM_LABELS - 1})"
        )
        df = df[df[label_column].between(0, NUM_LABELS - 1)]

    return df
