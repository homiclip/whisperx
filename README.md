# WhisperX API

REST API for audio transcription with [WhisperX](https://github.com/m-bain/whisperX): word-level timestamps and forced alignment (wav2vec2). **CPU only** (`int8`), no GPU.

## Build & Run

### Docker

```bash
docker build -t whisperx-api .
docker run -p 3000:3000 -p 5000:5000 whisperx-api
```

- **Port 3000** — API (`/transcribe`, `/docs`, etc.)
- **Port 5000** — Technical server: `GET /livez`, `GET /readyz`, `GET /metrics` (probes and Prometheus scrape). Override with `TECHNICAL_PORT`.

### Makefile (venv)

From the project root (creates/uses `.venv`). Run `make help` to list all targets.

| Target     | Description                                      |
|------------|--------------------------------------------------|
| `help`     | List all targets                                 |
| `venv`     | Create `.venv`                                   |
| `deps`     | Install/update deps (torch CPU + requirements)   |
| `deps-lint` | Install ruff (for lint); creates venv if needed  |
| `build`    | Check that `app.main` imports                    |
| `run`      | Run API on 3000 and technical server on 5000      |
| `lint`     | Run ruff on the code (requires `deps-lint`)      |
| `image`    | Build multi-arch Docker image (no push)          |
| `push`     | Build and push image to Docker Hub               |
| `release`  | Release new version                              | 

```bash
make help                     # list all targets
make deps && make run
make deps-lint && make lint   # install ruff and run lint
make image                    # build only
make push                     # build and push (DOCKER_REPOSITORY=kperreau/whisperx)
make push DOCKER_REPOSITORY=myuser/whisperx
make release
```

### _scripts/docker-build-push.sh

Builds `linux/amd64` with docker buildx. Options: `PUSH`, `DOCKER_REPOSITORY`, `PROJECT_PATH`.

```bash
PUSH=false  ./_scripts/docker-build-push.sh   # build only
PUSH=true   ./_scripts/docker-build-push.sh   # build and push
```

### CI (GitHub Actions)

`.github/workflows/docker.yml` builds and pushes the image on **push to `main`** and on **tags `v*`**.  
Uses **docker/setup-qemu-action** so `linux/arm64` can be built on amd64 runners (required for Apple Silicon / Mac M1–M3).

**Secrets:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`  
**Variable (optional):** `DOCKER_REPOSITORY` (default: `kperreau/whisperx`)

**Local multi-arch on Linux (Intel/amd64):** QEMU must be available to build `linux/arm64`, e.g.  
`docker run --privileged --rm tonistiigi/binfmt --install all`. Docker Desktop for Mac usually has it.

## Endpoints

Technical endpoints (livez, readyz, metrics) are on **port 5000**; the API is on **port 3000**.

| Port | Endpoints |
|------|-----------|
| **5000** | **GET /livez** — Liveness (Kubernetes `livenessProbe`). **GET /readyz** — Readiness (200 when model is loaded, 503 while loading). **GET /metrics** — Prometheus scrape. |
| **3000** | **POST /transcribe** — Upload an audio file (multipart). **GET /docs**, **GET /openapi.json**. |

Responses include an **X-Request-ID** header (or the one sent in the request) for tracing. When OpenTelemetry traces are enabled, responses also include **X-Trace-Id** (32-char hex), and the same `trace_id` is added to structured logs.

### POST /transcribe

- **Body**: `file` (wav, mp3, m4a, flac, ogg, webm, mp4)
- **Query**:
  - `language` (optional): `en`, `fr`, `de`, etc. Auto-detect if omitted.
  - `align` (default: `true`): enable word-level alignment.
- **Form**:
  - `initial_prompt` (optional): text to guide transcription (e.g. known script, domain vocabulary). Helps reduce errors when the audio content is known.

**Example:**

```bash
curl -X POST "http://localhost:3000/transcribe" -F "file=@audio.mp3"
curl -X POST "http://localhost:3000/transcribe?language=fr&align=true" -F "file=@voice.wav"
curl -X POST "http://localhost:3000/transcribe?language=en" -F "file=@output.mp3" -F "initial_prompt=Discover this beautiful house by the sea. It has 6 bedrooms and a bright living room. Contact me to visit."
```

**Response:**

```json
{
  "text": "Transcribed sentence...",
  "language": "fr",
  "segments": [
    { "start": 0.0, "end": 1.2, "text": "Phrase", "words": [...] }
  ]
}
```

## Environment variables

| Variable        | Default  | Description                                                                 |
|-----------------|----------|-----------------------------------------------------------------------------|
| `MODEL_NAME`    | `base`   | `tiny`, `base`, `small`, `medium`, `large-v2`. Use `base` or larger for quality. |
| `BATCH_SIZE`    | `8`      | Batch size (reduce if OOM).                                                |
| `LOG_FORMAT`    | `console`| `json` in prod for Grafana (Docker sets `json`).                            |
| `VAD_METHOD`    | `silero` | VAD method: `silero` (same as CLI `--vad_method silero`) or `pyannote`.    |
| `VAD_CHUNK_SIZE`| `60`     | Chunk size (seconds) for VAD merging.                                     |
| `VAD_ONSET`     | `0.35`    | VAD onset threshold.                                                        |
| `VAD_OFFSET`    | `0.25`  | VAD offset threshold.                                                       |
| `VAD_PAD_ONSET` | `0.2`    | Padding (seconds) before speech start (Silero VAD).                        |
| `VAD_PAD_OFFSET`| `0.2`    | Padding (seconds) after speech end (Silero VAD).                            |
| `ALIGN_PRELOAD_LANGUAGES` | *(empty)* | Comma-separated language codes (e.g. `en,fr`) to download alignment models (wav2vec2) at startup so `/readyz` is 200 only when everything is ready and no download happens on first transcribe. |
| `TECHNICAL_PORT` | `5000` | Port for the technical server (livez, readyz, metrics). API stays on 3000. |

**OpenTelemetry (telemetry)** — disabled by default. Traces are exported via OTLP (push). Metrics are exposed on `GET /metrics` (technical port) for Prometheus/Alloy scrape (pull). TraceContext + Baggage propagation; `trace_id` in logs and `X-Trace-Id` header.

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Master switch: enable OpenTelemetry (traces + metrics follow `OTEL_TRACES_ENABLED` / `OTEL_METRICS_ENABLED` unless overridden). |
| `OTEL_TRACES_ENABLED` | *(= OTEL_ENABLED)* | Enable trace export (OTLP push). When true, each request gets a span and `X-Trace-Id` is set; `trace_id` is added to structlog. |
| `OTEL_METRICS_ENABLED` | *(= OTEL_ENABLED)* | Enable metrics: expose Prometheus format on `GET /metrics` (pull; scrape with Prometheus/Alloy). |
| `OTEL_SERVICE_NAME` | `whisperx` | Service name for resource attributes. |
| `OTEL_SERVICE_VERSION` | `1.0.0` | Service version for resource attributes. |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Root sampler ratio (0.0–1.0). `1.0` = sample all; lower values reduce volume. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(SDK default)* | OTLP HTTP endpoint for traces only (e.g. `http://grafana-alloy:4318`). Metrics use `/metrics` (pull). |

Runs **CPU only** (`int8`); no GPU support.

## Production

- **Logs:** set `LOG_FORMAT=json` for structured JSON (timestamp, level, path, method, status_code, duration_ms, request_id, etc.) for Grafana/Loki. With `OTEL_TRACES_ENABLED=true`, each log line includes `trace_id` for correlation with traces.
- **Telemetry:** set `OTEL_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT` for traces (OTLP push). Metrics are exposed on `GET /metrics` on the technical port (5000); configure Prometheus/Alloy to scrape that endpoint. You can enable only traces or only metrics via `OTEL_TRACES_ENABLED` / `OTEL_METRICS_ENABLED`.
- **Recovery:** unhandled exceptions are caught and logged; the server returns 500 instead of crashing. SIGTERM/SIGINT are logged and delegated to uvicorn for graceful shutdown.

## Image

- Base: `python:3.12-slim-bookworm`
- PyTorch **CPU only** to keep size down.
- `--no-cache-dir` and `apt` cleanup for smaller layers.
