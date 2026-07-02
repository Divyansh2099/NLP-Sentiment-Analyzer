#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Build Docker images for the NLP Sentiment Analyzer
# Usage: bash scripts/build.sh
# ──────────────────────────────────────────────────────────────────
set -e

echo "🔨 Building Docker images for NLP Sentiment Analyzer..."
echo ""

docker-compose build

echo ""
echo "✅ Build complete!"
echo "   Run with:  docker-compose up"
echo "   API docs:  http://localhost:8000/docs"
echo "   UI:        http://localhost:8501"
