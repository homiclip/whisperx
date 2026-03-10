"""Prometheus metrics for the API (exposed on the technical server /metrics)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, REGISTRY

from app import config

# Counter used to compute http_requests_per_second (e.g. rate() in Prometheus Adapter).
# Same REGISTRY as technical_app /metrics so it is exposed when scraping.
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests by status code, method and path",
    ["status_code", "method", "path"],
    namespace=config.METRICS_NAMESPACE,
    registry=REGISTRY,
)

# Latency histogram (seconds). Buckets aligned with typical API response times.
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Request latency by status code, method and path",
    ["status_code", "method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    namespace=config.METRICS_NAMESPACE,
    registry=REGISTRY,
)


def _normalize_path(path: str) -> str:
    """Reduce cardinality: keep known paths, collapse the rest to /other."""
    if path in ("/", "/transcribe", "/health", "/docs", "/redoc", "/openapi.json"):
        return path
    return "/other"


def record_http_request(method: str, path: str, status_code: int) -> None:
    """Record one HTTP request and its latency (used by middleware)."""
    path = _normalize_path(path)
    status = str(status_code)
    HTTP_REQUESTS_TOTAL.labels(status_code=status, method=method, path=path).inc()


def record_http_latency(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    """Record request duration (used by middleware)."""
    path = _normalize_path(path)
    status = str(status_code)
    HTTP_REQUEST_DURATION_SECONDS.labels(status_code=status, method=method, path=path).observe(
        duration_seconds
    )
