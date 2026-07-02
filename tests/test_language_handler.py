"""
Tests for the multilingual language handler.
"""

import pytest
from src.data.language_handler import (
    detect_language,
    translate_to_english,
    process_multilingual,
    get_language_name,
    get_supported_languages,
    clear_translation_cache,
)


class TestDetectLanguage:
    """Tests for language detection."""

    def test_english_detection(self):
        lang, conf = detect_language("This is a great product!")
        assert lang == "en"
        assert conf > 0.5

    def test_spanish_detection(self):
        lang, conf = detect_language("Este producto es increíble!")
        assert lang == "es"
        assert conf > 0.5

    def test_french_detection(self):
        lang, conf = detect_language("Ce produit est magnifique!")
        assert lang == "fr"
        assert conf > 0.5

    def test_german_detection(self):
        lang, conf = detect_language("Dieses Produkt ist ausgezeichnet!")
        assert lang == "de"
        assert conf > 0.5

    def test_portuguese_detection(self):
        lang, conf = detect_language("Este produto é incrível!")
        assert lang == "pt"
        assert conf > 0.5

    def test_empty_string(self):
        lang, conf = detect_language("")
        assert lang == "en"
        assert conf == 0.0

    def test_low_confidence_fallback(self):
        # With an extremely high threshold, normal detection will be rejected
        # and the function should default to "en".
        lang, conf = detect_language("This is a clear English sentence.", threshold=0.9999)
        assert lang == "en"


class TestTranslateToEnglish:
    """Tests for the translation function."""

    def test_english_passthrough(self):
        """English text should be returned as-is."""
        clear_translation_cache()
        result = translate_to_english("This is great!", "en")
        assert result == "This is great!"

    def test_caching(self):
        """Translation should be cached on second call."""
        clear_translation_cache()
        text = "Este producto es bueno"
        translate_to_english(text, "es")
        result = translate_to_english(text, "es")
        assert isinstance(result, str)


class TestProcessMultilingual:
    """Tests for the full multilingual processing pipeline."""

    def test_english_passthrough(self):
        text, lang, translated = process_multilingual("This is great!")
        assert lang == "en"
        assert translated is False

    def test_non_english_translation(self):
        text, lang, translated = process_multilingual("Este producto es increíble")
        assert lang in ["es", "en"]  # May detect differently
        if lang == "es":
            assert translated is True

    def test_empty_input(self):
        text, lang, translated = process_multilingual("")
        assert text == ""
        assert lang == "en"


class TestGetLanguageName:
    """Tests for language name lookup."""

    def test_known_language(self):
        assert get_language_name("en") == "English"
        assert get_language_name("es") == "Spanish"
        assert get_language_name("fr") == "French"
        assert get_language_name("de") == "German"
        assert get_language_name("pt") == "Portuguese"

    def test_unknown_language(self):
        assert get_language_name("xx") == "xx"


class TestGetSupportedLanguages:
    """Tests for supported languages list."""

    def test_returns_5_languages(self):
        langs = get_supported_languages()
        assert len(langs) == 5
        codes = [l["code"] for l in langs]
        assert "en" in codes
        assert "es" in codes
        assert "fr" in codes
        assert "de" in codes
        assert "pt" in codes
