#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Start both API and UI services via Docker Compose
# Usage: bash scripts/start.sh
# ──────────────────────────────────────────────────────────────────
set -e

echo "🚀 Starting NLP Sentiment Analyzer services..."
echo ""

docker-compose up --build

echo ""
echo "🛑 Services stopped."
