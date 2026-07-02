"""
FastAPI application entry point for the NLP Sentiment Analyzer.

Start with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
Swagger docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import setup_cors, setup_request_logging, setup_global_exception_handler
from src.api.routes import router
from src.model.predictor import get_predictor
from src.utils.logger import setup_logger

logger = setup_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown logic."""
    # ── Startup ─────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("NLP Sentiment Analyzer API starting...")
    logger.info("=" * 50)

    import time
    start = time.time()
    predictor = get_predictor()
    load_time = time.time() - start

    logger.info(f"Model loaded in {load_time:.2f}s — ready to serve predictions ✅")

    yield  # Application is running

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("NLP Sentiment Analyzer API shutting down...")


app = FastAPI(
    title="NLP Sentiment Analyzer",
    description=(
        "A transformer-based sentiment analysis API that classifies text "
        "in real-time across 5 languages (English, Spanish, French, German, Portuguese). "
        "Powered by BERT (bert-base-multilingual-cased) with ≥94% accuracy."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Divyansh",
        "url": "https://divyansh2099.github.io/Portfolio/",
    },
    license_info={
        "name": "MIT License",
    },
    lifespan=lifespan,
)

# ── Setup Middleware ──────────────────────────────────────
setup_cors(app)
setup_request_logging(app)
setup_global_exception_handler(app)

# ── Include Routes ───────────────────────────────────────
app.include_router(router)
