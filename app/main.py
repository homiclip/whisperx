"""
WhisperX API — audio transcription with word-level timestamps.
Model and alignment loaded at startup for fast responses.
Production-ready: livez/readyz, signal handling, structured JSON logging.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
from dataclasses import replace
import signal
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

import torch

# PyTorch 2.6+ defaults to weights_only=True; pyannote/HF checkpoints need False when
# the caller omits it. Patch before importing whisperx so lightning_fabric's torch.load
# sees it (they use `torch.load`, not a cached reference).
_orig_torch_load = torch.load

def _torch_load_patched(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _orig_torch_load(*args, weights_only=weights_only, **kwargs)

torch.load = _torch_load_patched

import structlog  # noqa: E402
from structlog.stdlib import ProcessorFormatter  # noqa: E402

import whisperx  # noqa: E402
from fastapi import FastAPI, UploadFile, HTTPException, Query, Request, Form  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

# --- Config (env). CPU only, compute_type=int8. ---
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MODEL_NAME = os.getenv("MODEL_NAME", "tiny")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LOG_FORMAT = os.getenv("LOG_FORMAT", "console")  # "json" for prod / Grafana

# --- Global state (loaded at startup) ---
_model = None
_align_cache: dict[str, tuple] = {}  # lang -> (model_a, metadata)
_lock = asyncio.Lock()  # one transcription at a time on CPU

# Will be set after setup_logging()
log = None


def _setup_logging() -> None:
    """Configure structlog and root logger. JSON in prod for Grafana."""
    global log
    use_json = LOG_FORMAT.lower() == "json"

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        ProcessorFormatter.wrap_for_formatter,  # required when using ProcessorFormatter + stdlib BoundLogger
    ]
    if use_json:
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # No renderer in chain: event_dict goes to stdlib; ProcessorFormatter does final render
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib (uvicorn, etc.) and structlog through the same format
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
    formatter = ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=foreign,
    )
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(formatter)
    root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)  # UserWarning (torchaudio, pyannote, …) → root → our format

    # Route third-party loggers (whisperx, lightning, pyannote, …) through our formatter.
    # WhisperX: add our handler so its setup_logging() is never called (it would install its
    # own "%(asctime)s - %(name)s - %(levelname)s" format). Children (whisperx.asr, etc.) propagate here.
    _redirect_logger("whisperx", formatter, attach_handler=True)
    for _name in (
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "pytorch_lightning",
        "lightning_utilities",
        "pyannote",
        "pyannote.audio",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.asgi",
    ):
        _redirect_logger(_name, formatter, attach_handler=False)

    log = structlog.get_logger(__name__)


def _redirect_logger(
    name: str,
    formatter: logging.Formatter | None = None,
    *,
    attach_handler: bool,
) -> None:
    """Route a third-party logger through our ProcessorFormatter. If attach_handler, we add our
    handler and set propagate=False (use for whisperx so its setup_logging is not triggered).
    Otherwise we clear its handlers and set propagate=True so it uses the root."""
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


# Configure logging at import so uvicorn's "Started server process" / "Waiting for application
# startup" use our format (uvicorn loads the app before starting the server, so we run first).
_setup_logging()

_prev_signal_handlers: dict[int, object] = {}


def _install_signal_handlers() -> None:
    """Log SIGTERM/SIGINT and delegate to previous handler (e.g. uvicorn) for graceful shutdown."""

    def _handler(signum: int, frame: object) -> None:
        name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else str(signum)
        try:
            structlog.get_logger(__name__).info("signal_received", signal=signum, signal_name=name)
        except Exception:
            pass
        prev = _prev_signal_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        else:
            sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            _prev_signal_handlers[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not available on this platform / in this thread


def _load_align(lang: str) -> tuple:
    if lang in _align_cache:
        return _align_cache[lang]
    model_a, meta = whisperx.load_align_model(language_code=lang, device=DEVICE)
    _align_cache[lang] = (model_a, meta)
    return model_a, meta


def _transcribe_sync(path: str, language: str | None, align: bool, initial_prompt: str | None) -> dict:
    audio = whisperx.load_audio(path)

    old_opts = None
    if initial_prompt is not None:
        old_opts = _model.options
        _model.options = replace(old_opts, initial_prompt=initial_prompt)
    try:
        if language:
            result = _model.transcribe(audio, batch_size=BATCH_SIZE, language=language)
        else:
            result = _model.transcribe(audio, batch_size=BATCH_SIZE)
    finally:
        if old_opts is not None:
            _model.options = old_opts

    if align and result.get("segments"):
        model_a, meta = _load_align(result["language"])
        aligned = whisperx.align(
            result["segments"], model_a, meta, audio, DEVICE, return_char_alignments=False
        )
        result = {**result, **aligned}

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    # Re-apply uvicorn redirect in case it (re)added handlers after our import
    for _name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        _redirect_logger(_name, attach_handler=False)
    _install_signal_handlers()

    log.info("model_loading", model=MODEL_NAME, device=DEVICE)
    _model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
    log.info("model_loaded", model=MODEL_NAME, device=DEVICE)

    yield

    log.info("shutdown", message="cleaning up model and align cache")
    _model = None
    _align_cache.clear()
    gc.collect()


app = FastAPI(title="WhisperX API", version="1.0.0", lifespan=lifespan)


# --- Middleware: request_id, timing, structured logs, recover-like catch ---
@app.middleware("http")
async def _request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            "request",
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response
    except HTTPException:
        structlog.contextvars.clear_contextvars()
        raise
    except Exception as e:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.exception(
            "request_failed",
            path=request.url.path,
            method=request.method,
            duration_ms=duration_ms,
            error=str(e),
        )
        structlog.contextvars.clear_contextvars()
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id") or str(uuid.uuid4())
    log.exception("unhandled_exception", path=request.url.path, method=request.method, request_id=request_id)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id or str(uuid.uuid4())},
    )


# --- Liveness: process is up ---
@app.get("/livez")
def livez() -> dict:
    return {"status": "ok"}


# --- Readiness: model loaded and ready to transcribe ---
@app.get("/readyz")
def readyz() -> dict:
    if _model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model": MODEL_NAME, "message": "Model not loaded"},
        )
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    language: str | None = Query(None, description="Language code (en, fr, de, …). Auto-detect if omitted."),
    align: bool = Query(True, description="Enable word-level alignment (wav2vec2)."),
    initial_prompt: str | None = Form(None, description="Optional text to guide transcription (e.g. known script, domain vocabulary). Helps reduce errors when the audio content is known."),
) -> dict:
    if not file.filename or not any(
        file.filename.lower().endswith(ext) for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4")
    ):
        raise HTTPException(400, "Unsupported audio format. Use: wav, mp3, m4a, flac, ogg, webm, mp4.")

    suffix = os.path.splitext(file.filename)[1] or ".bin"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            chunk: bytes
            while chunk := await file.read(65536):
                tmp.write(chunk)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(500, f"Temporary write error: {e}")

    try:
        async with _lock:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, _transcribe_sync, tmp_path, language, align, initial_prompt or None)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    segments = out.get("segments", [])
    text = " ".join(s.get("text", "").strip() for s in segments).strip()
    return {
        "text": text,
        "language": out.get("language", ""),
        "segments": segments,
    }
