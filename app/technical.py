"""Technical server: livez, readyz, metrics on a dedicated port (default 5000)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from app import config
from app import model

technical_app = FastAPI(title="WhisperX technical", version="1.0.0", docs_url=None, redoc_url=None)


@technical_app.get("/livez")
def livez() -> dict:
    return {"status": "ok"}


@technical_app.get("/readyz")
def readyz() -> dict:
    if not model.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model": config.MODEL_NAME,
                "message": "Model not loaded",
            },
        )
    return {"status": "ok", "model": config.MODEL_NAME}


@technical_app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint (pull). OTel metrics when OTEL_METRICS_ENABLED."""
    from prometheus_client import REGISTRY, CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
