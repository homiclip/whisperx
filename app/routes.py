"""API routes: health, readiness, transcribe."""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app import config
from app import model

router = APIRouter()


@router.get("/livez")
def livez() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
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


@router.post("/transcribe")
async def transcribe(
    file: UploadFile,
    language: str | None = Query(None, description="Language code (en, fr, de, …). Auto-detect if omitted."),
    align: bool = Query(True, description="Enable word-level alignment (wav2vec2)."),
    initial_prompt: str | None = Form(None, description="Optional text to guide transcription."),
) -> dict:
    if not file.filename or not any(
        file.filename.lower().endswith(ext)
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4")
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
        out = await model.transcribe_async(tmp_path, language, align, initial_prompt)
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
