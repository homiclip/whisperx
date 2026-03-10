"""WhisperX model state, loading, and synchronous transcription."""

from __future__ import annotations

import asyncio
import gc
from contextlib import asynccontextmanager
from dataclasses import replace

import whisperx

from app import config
from app import logging_config

_model = None
_align_cache: dict[str, tuple] = {}
_lock = asyncio.Lock()


def get_model():
    """Return the loaded WhisperX model or None."""
    return _model


def is_ready() -> bool:
    return _model is not None


def _load_align(lang: str) -> tuple:
    if lang in _align_cache:
        return _align_cache[lang]
    model_a, meta = whisperx.load_align_model(language_code=lang, device=config.DEVICE)
    _align_cache[lang] = (model_a, meta)
    return model_a, meta


async def transcribe_async(
    path: str,
    language: str | None,
    align: bool,
    initial_prompt: str | None,
) -> dict:
    """Run transcribe_sync in executor under the global lock (one transcription at a time)."""
    loop = asyncio.get_running_loop()
    async with _lock:
        return await loop.run_in_executor(
            None,
            transcribe_sync,
            path,
            language,
            align,
            initial_prompt or None,
        )


def transcribe_sync(
    path: str,
    language: str | None,
    align: bool,
    initial_prompt: str | None,
) -> dict:
    """Run WhisperX transcribe (and optional align) on an audio file. Call from executor."""
    global _model
    audio = whisperx.load_audio(path)
    old_opts = None
    if initial_prompt is not None:
        old_opts = _model.options
        _model.options = replace(old_opts, initial_prompt=initial_prompt)
    try:
        if language:
            result = _model.transcribe(
                audio,
                batch_size=config.BATCH_SIZE,
                language=language,
                chunk_size=config.VAD_CHUNK_SIZE,
            )
        else:
            result = _model.transcribe(
                audio,
                batch_size=config.BATCH_SIZE,
                chunk_size=config.VAD_CHUNK_SIZE,
            )
    finally:
        if old_opts is not None:
            _model.options = old_opts
    if align and result.get("segments"):
        model_a, meta = _load_align(result["language"])
        aligned = whisperx.align(
            result["segments"], model_a, meta, audio, config.DEVICE, return_char_alignments=False
        )
        result = {**result, **aligned}
    return result


@asynccontextmanager
async def lifespan(app: object):
    global _model
    for _name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        logging_config._redirect_logger(_name, attach_handler=False)
    from app import signals
    signals.install_signal_handlers()

    log = logging_config.log
    log.info(
        "model_loading",
        model=config.MODEL_NAME,
        device=config.DEVICE,
        vad_method=config.VAD_METHOD,
        vad_chunk_size=config.VAD_CHUNK_SIZE,
        vad_onset=config.VAD_ONSET,
        vad_offset=config.VAD_OFFSET,
        vad_pad_onset=config.VAD_PAD_ONSET,
        vad_pad_offset=config.VAD_PAD_OFFSET,
    )
    _model = whisperx.load_model(
        config.MODEL_NAME,
        config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
        vad_method=config.VAD_METHOD,
        vad_options={
            "chunk_size": config.VAD_CHUNK_SIZE,
            "vad_onset": config.VAD_ONSET,
            "vad_offset": config.VAD_OFFSET,
            "pad_onset": config.VAD_PAD_ONSET,
            "pad_offset": config.VAD_PAD_OFFSET,
        },
    )
    log.info("model_loaded", model=config.MODEL_NAME, device=config.DEVICE)
    for lang in config.ALIGN_PRELOAD_LANGUAGES:
        try:
            _load_align(lang)
            log.info("align_model_preloaded", language=lang)
        except Exception as e:
            log.warning("align_model_preload_failed", language=lang, error=str(e))
    yield
    log.info("shutdown", message="cleaning up model and align cache")
    _model = None
    _align_cache.clear()
    gc.collect()
