"""HTTP middleware: request_id, trace_id, timing, structured logs."""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app import config
from app import metrics
from app import otel

# Use structlog.get_logger so we have a valid logger at request time (logging_config.log
# is only set after setup_logging() runs in main, which is after this module is imported).
log = structlog.get_logger(__name__)


async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_sec = time.perf_counter() - start
        metrics.record_http_request(request.method, request.url.path, response.status_code)
        metrics.record_http_latency(request.method, request.url.path, response.status_code, duration_sec)
        trace_id_hex = otel.get_trace_id_hex()
        duration_ms = round(duration_sec * 1000, 2)
        if request.url.path not in config.SKIP_REQUEST_LOG_PATHS:
            log.info(
                "request",
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
        response.headers["X-Request-ID"] = request_id
        if trace_id_hex:
            response.headers["X-Trace-Id"] = trace_id_hex
        structlog.contextvars.clear_contextvars()
        return response
    except HTTPException as exc:
        duration_sec = time.perf_counter() - start
        metrics.record_http_request(request.method, request.url.path, exc.status_code)
        metrics.record_http_latency(request.method, request.url.path, exc.status_code, duration_sec)
        structlog.contextvars.clear_contextvars()
        raise
    except Exception as e:
        duration_sec = time.perf_counter() - start
        metrics.record_http_request(request.method, request.url.path, 500)
        metrics.record_http_latency(request.method, request.url.path, 500, duration_sec)
        trace_id_hex = otel.get_trace_id_hex()
        duration_ms = round(duration_sec * 1000, 2)
        log.exception(
            "request_failed",
            path=request.url.path,
            method=request.method,
            duration_ms=duration_ms,
            error=str(e),
        )
        structlog.contextvars.clear_contextvars()
        out_headers = {"X-Request-ID": request_id}
        if trace_id_hex:
            out_headers["X-Trace-Id"] = trace_id_hex
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers=out_headers,
        )


async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id") or str(uuid.uuid4())
    trace_id_hex = otel.get_trace_id_hex()
    log.exception("unhandled_exception", path=request.url.path, method=request.method, request_id=request_id)
    headers = {}
    if trace_id_hex:
        headers["X-Trace-Id"] = trace_id_hex
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id or str(uuid.uuid4())},
        headers=headers,
    )
