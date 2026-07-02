# ──────────────────────────────────────────────────────────────────
# NLP Sentiment Analyzer — Multi-stage Dockerfile
# ──────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ───────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements and install to a target directory
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pytest.ini pyproject.toml .env.example README.md ./

# Create necessary directories
RUN mkdir -p data/raw data/processed models reports logs

# Default command (overridden by docker-compose per service)
CMD ["python", "scripts/run_api.py"]
