"""Signal handlers for graceful shutdown (SIGTERM/SIGINT)."""

from __future__ import annotations

import signal
import sys

import structlog

_prev_handlers: dict[int, object] = {}


def install_signal_handlers() -> None:
    """Log SIGTERM/SIGINT and delegate to previous handler (e.g. uvicorn) for graceful shutdown."""
    def _handler(signum: int, frame: object) -> None:
        name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else str(signum)
        try:
            structlog.get_logger(__name__).info("signal_received", signal=signum, signal_name=name)
        except Exception:
            pass
        prev = _prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        else:
            sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            _prev_handlers[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
