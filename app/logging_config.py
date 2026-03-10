"""Structlog and stdlib logging setup. Trace ID injected when OTEL traces are enabled."""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.stdlib import ProcessorFormatter

from app import config

# Trace ID processor uses opentelemetry.trace when OTEL is enabled
from opentelemetry import trace  # noqa: E402

log: structlog.stdlib.BoundLogger | None = None


def _add_trace_id_to_log(_: logging.Logger, __: str, event_dict: dict) -> dict:
    """Inject trace_id into every log line from current OTEL span when valid."""
    if config.OTEL_TRACES_ENABLED:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
    return event_dict


def _redirect_logger(
    name: str,
    formatter: logging.Formatter | None = None,
    *,
    attach_handler: bool,
) -> None:
    """Route a third-party logger through our ProcessorFormatter."""
    if attach_handler and formatter is None:
        raise TypeError("formatter required when attach_handler is True")
    logr = logging.getLogger(name)
    logr.handlers.clear()
    logr.setLevel(logging.NOTSET)
    if attach_handler:
        assert formatter is not None
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(formatter)
        logr.addHandler(h)
        logr.propagate = False
        logr.setLevel(logging.INFO)
    else:
        logr.propagate = True


def setup_logging() -> None:
    """Configure structlog and root logger. JSON in prod for Grafana."""
    global log
    use_json = config.LOG_FORMAT.lower() == "json"

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        _add_trace_id_to_log,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        ProcessorFormatter.wrap_for_formatter,
    ]
    if use_json:
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    def _add_logger_name(_: logging.Logger, __: str, event_dict: dict) -> dict:
        r = event_dict.get("_record")
        if r is not None and hasattr(r, "name"):
            event_dict["logger"] = r.name
        return event_dict

    foreign = [
        _add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    formatter = ProcessorFormatter(processor=renderer, foreign_pre_chain=foreign)

    class _SkipUvicornAccessFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.name != "uvicorn.access":
                return True
            msg = record.getMessage()
            return not any(path in msg for path in config.SKIP_REQUEST_LOG_PATHS)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    h.addFilter(_SkipUvicornAccessFilter())
    h.setFormatter(formatter)
    root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)

    _redirect_logger("whisperx", formatter, attach_handler=True)
    for _name in (
        "lightning", "lightning.pytorch", "lightning_fabric",
        "pytorch_lightning", "lightning_utilities",
        "pyannote", "pyannote.audio",
        "uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi",
    ):
        _redirect_logger(_name, formatter, attach_handler=False)

    log = structlog.get_logger(__name__)
