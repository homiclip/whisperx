"""
Audio loading: use pyannote (torchcodec) when available, else whisperx (FFmpeg subprocess).

Pyannote gives better decoding when torchcodec is installed (e.g. on Intel amd64 in prod).
On environments where torchcodec is missing or incompatible (e.g. some Mac setups),
we fall back to whisperx.load_audio (FFmpeg CLI). Both produce mono 16 kHz float32 for WhisperX.
"""

from __future__ import annotations

import numpy as np

# Sample rate expected by WhisperX transcribe/align pipeline
SAMPLE_RATE = 16000

_use_pyannote: bool | None = None
_pyannote_audio = None


def _detect_loader() -> bool:
    """Detect once whether torchcodec is available so we can use pyannote's decoder."""
    global _use_pyannote
    if _use_pyannote is not None:
        return _use_pyannote
    try:
        import torchcodec  # noqa: F401
        _use_pyannote = True
    except Exception:
        _use_pyannote = False
    return _use_pyannote


def _get_pyannote_audio():
    """Lazy init of pyannote Audio(sample_rate=16000, mono='downmix')."""
    global _pyannote_audio
    if _pyannote_audio is None:
        from pyannote.audio.core.io import Audio
        _pyannote_audio = Audio(sample_rate=SAMPLE_RATE, mono="downmix")
    return _pyannote_audio


def load_audio(path: str) -> np.ndarray:
    """
    Load audio file as mono 16 kHz float32 waveform for WhisperX.

    Uses pyannote (torchcodec) when available, otherwise whisperx.load_audio (FFmpeg).
    """
    if _detect_loader():
        audio_loader = _get_pyannote_audio()
        waveform, sr = audio_loader({"audio": path})
        # (channels, time) -> (time,) float32
        out = waveform.squeeze(0).cpu().float().numpy()
        if out.dtype != np.float32:
            out = out.astype(np.float32)
        return out

    import whisperx
    return whisperx.load_audio(path, sr=SAMPLE_RATE)


def get_loader_name() -> str:
    """Return which backend is used for loading ('pyannote' or 'whisperx')."""
    return "pyannote" if _detect_loader() else "whisperx"
