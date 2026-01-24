# WhisperX API — CPU only, lightweight
# slim + ffmpeg + PyTorch CPU + WhisperX
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json

# ffmpeg for whisperx.load_audio (mp3, wav, m4a, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU only (no CUDA)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# single worker to avoid duplicating model in memory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
