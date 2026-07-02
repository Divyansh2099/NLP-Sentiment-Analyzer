"""
Centralized structured logging for the NLP Sentiment Analyzer.
Logs to both console (stdout) and file simultaneously.
"""

import logging
import sys
from pathlib import Path

from src.utils.config import LOG_LEVEL, LOG_FILE, LOGS_DIR


def setup_logger(name: str = "sentiment_analyzer") -> logging.Logger:
    """Create and return a configured logger instance."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ─────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler ─────────────────────────────────────
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
