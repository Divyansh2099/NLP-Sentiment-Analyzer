#!/usr/bin/env python
"""
Start the Streamlit web UI.

Usage:
    python scripts/run_ui.py
    python scripts/run_ui.py --port 8501
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start the Sentiment Analyzer Web UI")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"🎨 Starting Sentiment Analyzer UI at http://{args.host}:{args.port}")

    sys.exit(subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/ui/app.py",
        f"--server.port={args.port}",
        f"--server.address={args.host}",
        "--browser.gatherUsageStats=false",
    ]).returncode)


if __name__ == "__main__":
    main()
