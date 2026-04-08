"""
server/app.py — Entry point wrapper for multi-mode deployment (uv run / openenv serve).

Imports the main FastAPI app from the project root and exposes a main()
function so the server can be launched via:
    uv run server            (pyproject.toml scripts entry)
    python -m server.app     (direct module execution)
    uvicorn server.app:app   (production uvicorn)
"""

import os
import sys

# Ensure the project root (parent of this directory) is on sys.path so that
# env.py, tasks.py, grader.py, etc. can be imported correctly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import app  # noqa: E402  (root app.py — FastAPI application)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """
    Start the Production Incident Response Simulator server.

    Called by:
        uv run server            — via [project.scripts] in pyproject.toml
        python -m server.app     — direct module invocation

    Args:
        host: Bind address (default: 0.0.0.0)
        port: TCP port (default: 7860 — required by HuggingFace Spaces)
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production Incident Response Simulator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
