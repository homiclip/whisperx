# ============== builder ==============
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Build dependencies (builder stage only)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a venv to be copied as-is into the final image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch CPU only (in venv)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# App dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============== runtime ==============
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    PATH="/opt/venv/bin:$PATH"

# ffmpeg only in runtime stage
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the ready-to-use venv
COPY --from=builder /opt/venv /opt/venv

# Copy app only
COPY app/ ./app/

EXPOSE 8000

# Single worker to avoid duplicating the model in memory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
