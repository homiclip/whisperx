"""Application configuration from environment."""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no", ""):
        return False
    return default


# Whisper / VAD
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MODEL_NAME = os.getenv("MODEL_NAME", "base")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
VAD_METHOD = os.getenv("VAD_METHOD", "silero")
VAD_CHUNK_SIZE = int(os.getenv("VAD_CHUNK_SIZE", "60"))
VAD_ONSET = float(os.getenv("VAD_ONSET", "0.35"))
VAD_OFFSET = float(os.getenv("VAD_OFFSET", "0.25"))
VAD_PAD_ONSET = float(os.getenv("VAD_PAD_ONSET", "0.2"))
VAD_PAD_OFFSET = float(os.getenv("VAD_PAD_OFFSET", "0.2"))

ALIGN_PRELOAD_LANGUAGES = [
    s.strip().lower()
    for s in os.getenv("ALIGN_PRELOAD_LANGUAGES", "").split(",")
    if s.strip()
]

# Logging
LOG_FORMAT = os.getenv("LOG_FORMAT", "console")

# OpenTelemetry (disabled by default)
OTEL_ENABLED = _env_bool("OTEL_ENABLED", False)
OTEL_TRACES_ENABLED = _env_bool("OTEL_TRACES_ENABLED", OTEL_ENABLED)
OTEL_METRICS_ENABLED = _env_bool("OTEL_METRICS_ENABLED", OTEL_ENABLED)
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "whisperx")
OTEL_SERVICE_VERSION = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
OTEL_TRACES_SAMPLER_ARG = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))

# Technical server (livez, readyz, metrics) — separate port for probes and scraping
TECHNICAL_PORT = int(os.getenv("TECHNICAL_PORT", "5000"))

HTTP_PORT = int(os.getenv("HTTP_PORT", "3000"))

# Paths excluded from request logs (probes, metrics, docs)
SKIP_REQUEST_LOG_PATHS = frozenset({
    "/livez", "/readyz", "/metrics", "/health",
    "/docs", "/redoc", "/openapi.json",
})
