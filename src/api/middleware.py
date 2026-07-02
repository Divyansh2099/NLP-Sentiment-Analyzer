"""
Middleware for the Sentiment Analysis API.
Includes CORS configuration, request logging, and rate limiting.
"""

import time
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.config import RATE_LIMIT_PER_MINUTE
from src.utils.logger import setup_logger

logger = setup_logger("api.middleware")

# ── Simple In-Memory Rate Limiter ─────────────────────────
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_WINDOW_SECONDS = 60


def _check_rate_limit(client_ip: str) -> bool:
    """Check if the client has exceeded the rate limit.

    Returns:
        True if the request should be allowed, False if rate limited.
    """
    now = time.time()
    # Clean old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip]
        if now - t < RATE_WINDOW_SECONDS
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        return False
    _rate_limit_store[client_ip].append(now)
    return True


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware.

    Args:
        app: FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all in dev; restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware configured (allow_all for development)")


def setup_request_logging(app: FastAPI) -> None:
    """Add request/response logging middleware.

    Args:
        app: FastAPI application instance.
    """
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"

        # Rate limit check (skip health endpoint)
        if request.url.path != "/api/v1/health":
            if not _check_rate_limit(client_ip):
                logger.warning(f"Rate limited: {client_ip} on {request.url.path}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limited",
                        "detail": f"Too many requests. Limit is {RATE_LIMIT_PER_MINUTE}/min.",
                        "status_code": 429,
                    },
                )

        response: Response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        # Log predict calls
        if "/predict" in request.url.path and request.method == "POST":
            logger.info(
                f"{request.method} {request.url.path} → {response.status_code} "
                f"({duration_ms:.1f}ms) from {client_ip}"
            )

        return response

    logger.info("Request logging middleware configured")


def setup_global_exception_handler(app: FastAPI) -> None:
    """Add global exception handler for uncaught errors.

    Args:
        app: FastAPI application instance.
    """
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred. Please try again.",
                "status_code": 500,
            },
        )

    logger.info("Global exception handler configured")
