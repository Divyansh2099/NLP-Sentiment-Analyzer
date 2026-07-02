"""
Application configuration loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if it exists
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# ── Project Paths ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / os.getenv("REPORTS_DIR", "reports")
LOGS_DIR = PROJECT_ROOT / "logs"

# ── Model Configuration ─────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-multilingual-cased")
MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_classifier")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_LABELS = int(os.getenv("NUM_LABELS", "3"))
LABELS = [l.strip() for l in os.getenv("LABELS", "negative,neutral,positive").split(",")]

# ── Training Hyperparameters ────────────────────────────
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "500"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))

# ── API Configuration ───────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# ── UI Configuration ─────────────────────────────────────
UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
UI_PORT = int(os.getenv("UI_PORT", "8501"))

# ── Language Configuration ──────────────────────────────
SUPPORTED_LANGUAGES = [l.strip() for l in os.getenv(
    "SUPPORTED_LANGUAGES", "en,es,fr,de,pt"
).split(",")]
LANGUAGE_CONFIDENCE_THRESHOLD = float(os.getenv("LANGUAGE_CONFIDENCE_THRESHOLD", "0.8"))

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
}

# ── Logging ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/sentiment_analyzer.log")

# ── Ensure directories exist ────────────────────────────
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
