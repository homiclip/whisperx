# WhisperX API

REST API for audio transcription with [WhisperX](https://github.com/m-bain/whisperX): word-level timestamps and forced alignment (wav2vec2). **CPU only** (`int8`), no GPU.

## Build & Run

### Docker

```bash
docker build -t whisperx-api .
docker run -p 8000:8000 whisperx-api
```

### Makefile (venv)

From the project root (creates/uses `.venv`). Run `make help` to list all targets.

| Target     | Description                                      |
|------------|--------------------------------------------------|
| `help`     | List all targets                                 |
| `venv`     | Create `.venv`                                   |
| `deps`     | Install/update deps (torch CPU + requirements)   |
| `deps-lint` | Install ruff (for lint); creates venv if needed  |
| `build`    | Check that `app.main` imports                    |
| `run`      | Run uvicorn on `0.0.0.0:8000`                    |
| `lint`     | Run ruff on the code (requires `deps-lint`)      |
| `image`    | Build multi-arch Docker image (no push)          |
| `push`     | Build and push image to Docker Hub               |

```bash
make help                     # list all targets
make deps && make run
make deps-lint && make lint   # install ruff and run lint
make image                    # build only
make push                     # build and push (DOCKER_REPOSITORY=kperreau/whisperx)
make push DOCKER_REPOSITORY=myuser/whisperx
```

### scripts/docker-build-push.sh

Builds `linux/amd64` and `linux/arm64` with docker buildx. Options: `PUSH`, `DOCKER_REPOSITORY`, `PROJECT_PATH`.

```bash
PUSH=false  ./scripts/docker-build-push.sh   # build only
PUSH=true   ./scripts/docker-build-push.sh   # build and push
```

### CI (GitHub Actions)

`.github/workflows/docker.yml` builds and pushes the image on **push to `main`** and on **tags `v*`**.  
Uses **docker/setup-qemu-action** so `linux/arm64` can be built on amd64 runners (required for Apple Silicon / Mac M1–M3).

**Secrets:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`  
**Variable (optional):** `DOCKER_REPOSITORY` (default: `kperreau/whisperx`)

**Local multi-arch on Linux (Intel/amd64):** QEMU must be available to build `linux/arm64`, e.g.  
`docker run --privileged --rm tonistiigi/binfmt --install all`. Docker Desktop for Mac usually has it.

## Endpoints

- **GET /livez** — Liveness: process is up. Use for Kubernetes `livenessProbe`.
- **GET /readyz** — Readiness: returns 200 only when the model is loaded and ready to transcribe. Use for `readinessProbe`. Returns 503 while loading or if the model is unavailable.
- **POST /transcribe** — Upload an audio file (multipart)

Responses include an **X-Request-ID** header (or the one sent in the request) for tracing.

### POST /transcribe

- **Body**: `file` (wav, mp3, m4a, flac, ogg, webm, mp4)
- **Query**:
  - `language` (optional): `en`, `fr`, `de`, etc. Auto-detect if omitted.
  - `align` (default: `true`): enable word-level alignment.
- **Form**:
  - `initial_prompt` (optional): text to guide transcription (e.g. known script, domain vocabulary). Helps reduce errors when the audio content is known.

**Example:**

```bash
curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.mp3"
curl -X POST "http://localhost:8000/transcribe?language=fr&align=true" -F "file=@voice.wav"
curl -X POST "http://localhost:8000/transcribe?language=en" -F "file=@output.mp3" -F "initial_prompt=Discover this beautiful house by the sea. It has 6 bedrooms and a bright living room. Contact me to visit."
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
| `VAD_CHUNK_SIZE`| `20`     | Chunk size (seconds) for VAD merging.                                     |
| `VAD_ONSET`     | `0.35`    | VAD onset threshold.                                                        |
| `VAD_OFFSET`    | `0.25`  | VAD offset threshold.                                                       |
| `VAD_PAD_ONSET` | `0.2`    | Padding (seconds) before speech start (Silero VAD).                        |
| `VAD_PAD_OFFSET`| `0.2`    | Padding (seconds) after speech end (Silero VAD).                            |

Runs **CPU only** (`int8`); no GPU support.

## Production

- **Logs:** set `LOG_FORMAT=json` for structured JSON (timestamp, level, path, method, status_code, duration_ms, request_id, etc.) for Grafana/Loki.
- **Recovery:** unhandled exceptions are caught and logged; the server returns 500 instead of crashing. SIGTERM/SIGINT are logged and delegated to uvicorn for graceful shutdown.

## Image

- Base: `python:3.12-slim-bookworm`
- PyTorch **CPU only** to keep size down.
- `--no-cache-dir` and `apt` cleanup for smaller layers.
