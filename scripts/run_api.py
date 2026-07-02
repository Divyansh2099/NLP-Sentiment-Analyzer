#!/usr/bin/env python
"""
Start the FastAPI server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 9000 --host 127.0.0.1
"""

import argparse
import uvicorn

from src.utils.config import API_HOST, API_PORT


def main():
    parser = argparse.ArgumentParser(description="Start the Sentiment Analysis API")
    parser.add_argument("--host", type=str, default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"🚀 Starting Sentiment Analysis API at http://{args.host}:{args.port}")
    print(f"   Swagger docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
