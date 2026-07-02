"""
Tests for the application configuration.

Validates that config values load correctly and have sensible defaults.
"""

from src.utils import config


class TestProjectPaths:
    """Tests for project path constants."""

    def test_project_root_exists(self):
        assert config.PROJECT_ROOT.exists()

    def test_data_directories(self):
        assert config.DATA_RAW_DIR.exists()
        assert config.DATA_PROCESSED_DIR.exists()

    def test_models_dir_exists(self):
        assert config.MODELS_DIR.exists()

    def test_reports_dir_exists(self):
        assert config.REPORTS_DIR.exists()


class TestModelConfig:
    """Tests for model configuration."""

    def test_model_name(self):
        assert config.MODEL_NAME == "bert-base-multilingual-cased"

    def test_num_labels(self):
        assert config.NUM_LABELS == 3

    def test_labels_order(self):
        # Order matters: 0=negative, 1=neutral, 2=positive
        assert config.LABELS == ["negative", "neutral", "positive"]

    def test_max_seq_length(self):
        assert 64 <= config.MAX_SEQ_LENGTH <= 1024


class TestTrainingConfig:
    """Tests for training hyperparameters."""

    def test_learning_rate_range(self):
        assert 1e-6 < config.LEARNING_RATE < 1e-3

    def test_epochs_positive(self):
        assert config.EPOCHS >= 1

    def test_batch_size_positive(self):
        assert config.BATCH_SIZE >= 1


class TestLanguageConfig:
    """Tests for language configuration."""

    def test_five_supported_languages(self):
        assert len(config.SUPPORTED_LANGUAGES) == 5

    def test_english_first(self):
        assert "en" in config.SUPPORTED_LANGUAGES

    def test_all_languages_have_names(self):
        for code in config.SUPPORTED_LANGUAGES:
            assert code in config.LANGUAGE_NAMES

    def test_language_names_mapping(self):
        assert config.LANGUAGE_NAMES["en"] == "English"
        assert config.LANGUAGE_NAMES["es"] == "Spanish"
        assert config.LANGUAGE_NAMES["fr"] == "French"
        assert config.LANGUAGE_NAMES["de"] == "German"
        assert config.LANGUAGE_NAMES["pt"] == "Portuguese"

    def test_confidence_threshold_range(self):
        assert 0.0 <= config.LANGUAGE_CONFIDENCE_THRESHOLD <= 1.0
