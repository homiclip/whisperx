"""
WhisperX API — audio transcription with word-level timestamps.
Model and alignment loaded at startup. Production-ready: livez/readyz, signals, structlog, optional OTEL.
"""

from __future__ import annotations

import warnings

import torch

# PyTorch 2.6+ defaults to weights_only=True; pyannote/HF checkpoints need False.
# Patch before importing whisperx so lightning_fabric's torch.load sees it.
_orig_torch_load = torch.load


def _torch_load_patched(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _orig_torch_load(*args, weights_only=weights_only, **kwargs)


torch.load = _torch_load_patched

# Suppress pyannote/torchcodec UserWarning when torchcodec is missing (we fall back to whisperx).
# Match by module: message starts with \n so a message regex often fails.
warnings.filterwarnings("ignore", module="pyannote.audio.core.io", category=UserWarning)

import whisperx  # noqa: E402, F401  (side effect: load after torch patch)

from fastapi import FastAPI  # noqa: E402

from app import logging_config  # noqa: E402
from app import middleware  # noqa: E402
from app import model  # noqa: E402
from app import otel  # noqa: E402
from app import routes  # noqa: E402

# Configure logging at import so uvicorn startup messages use our format.
logging_config.setup_logging()

app = FastAPI(title="WhisperX API", version="1.0.0", lifespan=model.lifespan)

otel.setup_otel(app)
app.middleware("http")(middleware.request_logging_middleware)
app.add_exception_handler(Exception, middleware.global_exception_handler)
app.include_router(routes.router)
