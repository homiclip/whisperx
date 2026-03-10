"""
Run API on port 3000 and technical server (livez, readyz, metrics) on port 5000.
Both run in the same process so readiness reflects the shared model state.
"""

from __future__ import annotations

import threading

import uvicorn

from app import config

# Import main app first so OTEL and model lifespan are configured
from app.main import app as main_app
from app.technical import technical_app


def _run_technical() -> None:
    uvicorn.run(
        technical_app,
        host="0.0.0.0",
        port=config.TECHNICAL_PORT,
        log_config=None,
    )


def main() -> None:
    thread = threading.Thread(target=_run_technical, daemon=True)
    thread.start()
    uvicorn.run(
        main_app,
        host="0.0.0.0",
        port=config.HTTP_PORT,
    )


if __name__ == "__main__":
    main()
