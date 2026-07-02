"""
Tests for the dataset builder — combine, balance, and split logic.

These tests run without torch/transformers. They use small synthetic
DataFrames to validate the building pipeline in isolation.
"""

import pandas as pd
import pytest

from src.data.dataset_builder import (
    balance_classes,
    combine_datasets,
    split_dataset,
)


@pytest.fixture
def sample_datasets():
    """Two small datasets with clean_text + label + source columns."""
    df1 = pd.DataFrame({
        "clean_text": [f"great review {i}" for i in range(30)] +
                      [f"bad review {i}" for i in range(10)],
        "label": [2] * 30 + [0] * 10,
        "source": ["source_a"] * 40,
    })
    df2 = pd.DataFrame({
        "clean_text": [f"okay text {i}" for i in range(20)],
        "label": [1] * 20,
        "source": ["source_b"] * 20,
    })
    return {"source_a": df1, "source_b": df2}


class TestCombineDatasets:
    """Tests for combine_datasets."""

    def test_combines_all_rows(self, sample_datasets):
        combined = combine_datasets(sample_datasets)
        assert len(combined) == 40 + 20

    def test_keeps_required_columns(self, sample_datasets):
        combined = combine_datasets(sample_datasets)
        assert "clean_text" in combined.columns
        assert "label" in combined.columns
        assert "source" in combined.columns

    def test_skips_missing_clean_text(self, sample_datasets):
        # Add a dataset without clean_text column
        bad = {"bad_source": pd.DataFrame({"text": ["x"], "label": [0]})}
        result = combine_datasets({**sample_datasets, **bad})
        # Should still include the valid datasets, skip the bad one
        assert len(result) == 60

    def test_adds_source_if_missing(self, sample_datasets):
        df_no_source = pd.DataFrame({
            "clean_text": ["hello world"],
            "label": [2],
        })
        combined = combine_datasets({"no_source": df_no_source})
        assert combined["source"].iloc[0] == "no_source"


class TestBalanceClasses:
    """Tests for balance_classes."""

    def test_balances_to_min_class(self, sample_datasets):
        combined = combine_datasets(sample_datasets)
        # Class counts: 0→10, 1→20, 2→30. Min is 10.
        balanced = balance_classes(combined)
        counts = balanced["label"].value_counts()
        assert counts[0] == 10
        assert counts[1] == 10
        assert counts[2] == 10

    def test_respects_target_per_class(self, sample_datasets):
        combined = combine_datasets(sample_datasets)
        balanced = balance_classes(combined, target_per_class=5)
        counts = balanced["label"].value_counts()
        assert counts[0] == 5
        assert counts[1] == 5
        assert counts[2] == 5

    def test_target_larger_than_class_keeps_all(self, sample_datasets):
        combined = combine_datasets(sample_datasets)
        # target=1000 is larger than all class counts → keeps everything per class
        balanced = balance_classes(combined, target_per_class=1000)
        counts = balanced["label"].value_counts()
        assert counts[0] == 10
        assert counts[1] == 20
        assert counts[2] == 30


class TestSplitDataset:
    """Tests for split_dataset (stratified train/val/test)."""

    def test_total_size_preserved(self):
        df = pd.DataFrame({
            "clean_text": [f"text {i}" for i in range(300)],
            "label": [0] * 100 + [1] * 100 + [2] * 100,
        })
        train, val, test = split_dataset(df, test_size=0.1, val_size=0.1)
        assert len(train) + len(val) + len(test) == 300

    def test_approximate_split_ratios(self):
        df = pd.DataFrame({
            "clean_text": [f"text {i}" for i in range(1000)],
            "label": [0] * 334 + [1] * 333 + [2] * 333,
        })
        train, val, test = split_dataset(df, test_size=0.1, val_size=0.1)
        # 80/10/10 ± a few samples
        assert 780 <= len(train) <= 820
        assert 80 <= len(val) <= 120
        assert 80 <= len(test) <= 120

    def test_splits_are_disjoint(self):
        df = pd.DataFrame({
            "clean_text": [f"text {i}" for i in range(300)],
            "label": [0] * 100 + [1] * 100 + [2] * 100,
        })
        train, val, test = split_dataset(df)
        train_set = set(train["clean_text"])
        val_set = set(val["clean_text"])
        test_set = set(test["clean_text"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_stratification_preserves_classes(self):
        df = pd.DataFrame({
            "clean_text": [f"text {i}" for i in range(300)],
            "label": [0] * 100 + [1] * 100 + [2] * 100,
        })
        train, val, test = split_dataset(df)
        # Each split should contain all 3 classes
        for split in [train, val, test]:
            assert sorted(split["label"].unique()) == [0, 1, 2]
