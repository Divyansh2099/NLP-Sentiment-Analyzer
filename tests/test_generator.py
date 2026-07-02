"""
Tests for the synthetic dataset generator.

These tests run without torch/transformers — the generator is pure Python
plus pandas. They validate that generated data has the right structure,
label distribution, and language coverage.
"""

import pandas as pd
import pytest

from src.data.generator import (
    ReviewGenerator,
    DOMAINS,
    generate_full_dataset,
    generate_multilingual_samples,
)


class TestReviewGenerator:
    """Tests for the ReviewGenerator class."""

    def setup_method(self):
        import random
        self.gen = ReviewGenerator(rng=random.Random(42))

    def test_generates_positive_review(self):
        review = self.gen.generate_review("positive", "electronics", noise_level=0.0)
        assert isinstance(review, str)
        assert len(review) > 10

    def test_generates_neutral_review(self):
        review = self.gen.generate_review("neutral", "hotel", noise_level=0.0)
        assert isinstance(review, str)
        assert len(review) > 10

    def test_generates_negative_review(self):
        review = self.gen.generate_review("negative", "restaurant", noise_level=0.0)
        assert isinstance(review, str)
        assert len(review) > 10

    def test_all_domains_produce_valid_output(self):
        for domain in DOMAINS:
            for sentiment in ["positive", "neutral", "negative"]:
                review = self.gen.generate_review(sentiment, domain, noise_level=0.0)
                assert isinstance(review, str)
                assert len(review) > 5

    def test_invalid_sentiment_raises(self):
        with pytest.raises(KeyError):
            self.gen.generate_review("unknown", "electronics")

    def test_invalid_domain_raises(self):
        with pytest.raises(KeyError):
            self.gen.generate_review("positive", "unknown_domain")

    def test_reproducible_with_seed(self):
        import random
        gen1 = ReviewGenerator(rng=random.Random(123))
        gen2 = ReviewGenerator(rng=random.Random(123))
        r1 = gen1.generate_review("positive", "electronics", noise_level=0.0)
        r2 = gen2.generate_review("positive", "electronics", noise_level=0.0)
        assert r1 == r2


class TestGenerateBatch:
    """Tests for generate_batch."""

    def test_returns_dataframe(self):
        gen = ReviewGenerator()
        df = gen.generate_batch(n_per_class=10, noise_level=0.0)
        assert isinstance(df, pd.DataFrame)

    def test_balanced_classes(self):
        gen = ReviewGenerator()
        df = gen.generate_batch(n_per_class=50, noise_level=0.0)
        counts = df["label"].value_counts().sort_index()
        assert counts[0] == 50  # negative
        assert counts[1] == 50  # neutral
        assert counts[2] == 50  # positive

    def test_total_count(self):
        gen = ReviewGenerator()
        df = gen.generate_batch(n_per_class=20, noise_level=0.0)
        assert len(df) == 60

    def test_has_required_columns(self):
        gen = ReviewGenerator()
        df = gen.generate_batch(n_per_class=5, noise_level=0.0)
        assert "text" in df.columns
        assert "label" in df.columns
        assert "source" in df.columns

    def test_labels_in_valid_range(self):
        gen = ReviewGenerator()
        df = gen.generate_batch(n_per_class=10, noise_level=0.0)
        assert df["label"].min() >= 0
        assert df["label"].max() <= 2


class TestGenerateMultilingual:
    """Tests for generate_multilingual_samples."""

    def test_returns_dataframe(self):
        df = generate_multilingual_samples(n_per_lang_per_class=5, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_count(self):
        # 4 languages × 3 classes × n_per_lang_per_class
        df = generate_multilingual_samples(n_per_lang_per_class=10, seed=42)
        assert len(df) == 4 * 3 * 10

    def test_has_required_columns(self):
        df = generate_multilingual_samples(n_per_lang_per_class=2, seed=42)
        assert "text" in df.columns
        assert "label" in df.columns
        assert "source" in df.columns

    def test_source_includes_language_codes(self):
        df = generate_multilingual_samples(n_per_lang_per_class=2, seed=42)
        sources = df["source"].unique()
        for lang in ["es", "fr", "de", "pt"]:
            assert any(lang in s for s in sources)


class TestGenerateFullDataset:
    """Tests for generate_full_dataset."""

    def test_combines_english_and_multilingual(self):
        df = generate_full_dataset(
            n_english_per_class=50,
            n_multilingual_per_class_per_lang=10,
            noise_level=0.0,
        )
        assert isinstance(df, pd.DataFrame)
        # English: 50 × 3 = 150, Multilingual: 10 × 4 × 3 = 120
        assert len(df) == 150 + 120

    def test_balanced_classes(self):
        df = generate_full_dataset(
            n_english_per_class=30,
            n_multilingual_per_class_per_lang=5,
            noise_level=0.0,
        )
        counts = df["label"].value_counts()
        # Negative: 30 + 20 = 50
        assert counts[0] == 50
        assert counts[1] == 50
        assert counts[2] == 50

    def test_without_multilingual(self):
        df = generate_full_dataset(
            n_english_per_class=20,
            include_multilingual=False,
            noise_level=0.0,
        )
        assert len(df) == 60  # 20 × 3

    def test_noise_does_not_change_count(self):
        df_clean = generate_full_dataset(
            n_english_per_class=20, noise_level=0.0, include_multilingual=False
        )
        df_noisy = generate_full_dataset(
            n_english_per_class=20, noise_level=0.9, include_multilingual=False
        )
        assert len(df_clean) == len(df_noisy)
