"""
Multilingual language handler — detects language and translates non-English
text to English before sentiment analysis.

Supports 5 languages: English, Spanish, French, German, Portuguese.
Strategy: langdetect → deep-translator (with caching) → BERT inference.
"""

from functools import lru_cache

from src.utils.config import SUPPORTED_LANGUAGES, LANGUAGE_CONFIDENCE_THRESHOLD, LANGUAGE_NAMES
from src.utils.logger import setup_logger

logger = setup_logger("data.language_handler")

# ── Translation Cache ────────────────────────────────────
_translation_cache: dict[str, str] = {}


def detect_language(text: str, threshold: float = LANGUAGE_CONFIDENCE_THRESHOLD) -> tuple[str, float]:
    """Detect the language of the given text.

    Args:
        text: Input text string.
        threshold: Minimum confidence to accept detection (0-1).

    Returns:
        (language_code, confidence) — e.g., ("en", 0.99).
        Falls back to "en" with 0.0 confidence if detection fails.
    """
    if not text or not text.strip():
        return "en", 0.0

    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        top = results[0]
        lang_code = top.lang.lower()
        confidence = top.prob

        # Map common close variants
        code_map = {"zh-cn": "zh", "zh-tw": "zh"}
        lang_code = code_map.get(lang_code, lang_code)

        if confidence < threshold:
            logger.debug(
                f"Low confidence detection: {lang_code} ({confidence:.2f}) "
                f"below threshold {threshold} — defaulting to 'en'"
            )
            return "en", confidence

        return lang_code, confidence

    except Exception as e:
        logger.debug(f"Language detection failed: {e} — defaulting to 'en'")
        return "en", 0.0


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text from source language to English.

    Uses Google Translate with an in-memory cache to avoid re-translating
    identical strings.

    Args:
        text: Source text in any supported language.
        source_lang: ISO 639-1 language code of the source text.

    Returns:
        English translation of the text. Returns original text if already
        English or if translation fails.
    """
    # Skip translation for English
    if source_lang == "en":
        return text

    # Check cache
    cache_key = f"{source_lang}:{text}"
    if cache_key in _translation_cache:
        logger.debug(f"Translation cache hit for '{text[:50]}...'")
        return _translation_cache[cache_key]

    # Translate
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)

        # Guard against empty or None translation
        if not translated or not translated.strip():
            logger.warning(
                f"Empty translation for '{text[:50]}...' — returning original"
            )
            return text

        # Cache the result
        _translation_cache[cache_key] = translated
        logger.debug(
            f"Translated ({source_lang}→en): '{text[:50]}...' → "
            f"'{translated[:50]}...'"
        )
        return translated

    except Exception as e:
        logger.warning(f"Translation failed for '{text[:50]}...': {e}")
        return text


def process_multilingual(
    text: str,
    confidence_threshold: float = LANGUAGE_CONFIDENCE_THRESHOLD,
) -> tuple[str, str, bool]:
    """Full multilingual processing: detect → translate (if needed).

    Args:
        text: Input text in any supported language.
        confidence_threshold: Minimum confidence for language detection.

    Returns:
        (processed_text, detected_language, was_translated)
            - processed_text: English text ready for BERT
            - detected_language: ISO 639-1 code
            - was_translated: True if text was translated from another language
    """
    detected_lang, confidence = detect_language(text, threshold=confidence_threshold)

    # Only process supported languages; translate non-English
    was_translated = False
    processed_text = text

    if detected_lang != "en" and detected_lang in SUPPORTED_LANGUAGES:
        processed_text = translate_to_english(text, detected_lang)
        was_translated = True
    elif detected_lang not in SUPPORTED_LANGUAGES:
        # Unsupported language — try translating anyway
        logger.info(
            f"Unsupported language '{detected_lang}' — "
            f"attempting translation to English"
        )
        processed_text = translate_to_english(text, detected_lang)
        was_translated = True

    return processed_text, detected_lang, was_translated


def clear_translation_cache() -> None:
    """Clear the in-memory translation cache."""
    global _translation_cache
    _translation_cache.clear()
    logger.info("Translation cache cleared")


def get_language_name(code: str) -> str:
    """Convert language code to full name.

    Args:
        code: ISO 639-1 language code.

    Returns:
        Full language name (e.g., "Spanish") or the code if not found.
    """
    return LANGUAGE_NAMES.get(code, code)


def get_supported_languages() -> list[dict]:
    """Return list of supported languages with code and name.

    Returns:
        List of dicts: [{"code": "en", "name": "English"}, ...]
    """
    return [
        {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
        for code in SUPPORTED_LANGUAGES
    ]
