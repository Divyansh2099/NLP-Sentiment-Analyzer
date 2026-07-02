"""
Tests for the text preprocessing pipeline.
"""

import pytest
from src.data.preprocessor import clean_text, preprocess_dataframe
import pandas as pd


class TestCleanText:
    """Tests for the clean_text function."""

    def test_basic_cleaning(self):
        assert clean_text("Hello World!") == "hello world!"

    def test_html_removal(self):
        result = clean_text("<p>Hello</p><br>World</br>")
        assert "<p>" not in result
        assert "hello" in result

    def test_url_removal(self):
        result = clean_text("Check out https://example.com/page for more")
        assert "https://" not in result
        assert "example.com" not in result

    def test_mention_removal(self):
        result = clean_text("Hey @user great job!")
        assert "@" not in result
        assert "great job!" in result

    def test_hashtag_word_preserved(self):
        result = clean_text("This is #amazing")
        assert "amazing" in result

    def test_emoji_removal(self):
        result = clean_text("I love it! 😊🎉")
        assert "😊" not in result
        assert "i love it!" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_handling(self):
        assert clean_text(None) == ""

    def test_whitespace_normalization(self):
        result = clean_text("Hello    World   !  ")
        assert "    " not in result
        assert result == "hello world !"

    def test_long_text_truncation_handled(self):
        long_text = "a" * 1000 + " important"
        result = clean_text(long_text)
        assert len(result) > 0

    def test_html_entities(self):
        result = clean_text("Hello &amp; World &lt;test&gt;")
        assert "&amp;" not in result
        assert "hello & world" in result or "hello" in result

    def test_lowercase(self):
        result = clean_text("HELLO WORLD")
        assert result == "hello world"


class TestPreprocessDataFrame:
    """Tests for the preprocess_dataframe function."""

    def test_basic_preprocessing(self):
        df = pd.DataFrame({
            "text": ["Great product!", "Terrible service.", "It was okay."],
            "label": [2, 0, 1],
        })
        result = preprocess_dataframe(df)
        assert len(result) > 0
        assert "clean_text" in result.columns
        assert "great product!" in result["clean_text"].iloc[0]

    def test_removes_nulls(self):
        df = pd.DataFrame({
            "text": ["Good!", None, "Bad!", float("nan")],
            "label": [2, 1, 0, 0],
        })
        result = preprocess_dataframe(df)
        assert result["clean_text"].notna().all()

    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "text": ["Same text", "Same text", "Different text"],
            "label": [2, 2, 0],
        })
        result = preprocess_dataframe(df)
        assert len(result) < len(df)

    def test_min_length_filter(self):
        df = pd.DataFrame({
            "text": ["ab", "Very long text here"],
            "label": [2, 0],
        })
        result = preprocess_dataframe(df, min_text_length=3)
        assert len(result) == 1
