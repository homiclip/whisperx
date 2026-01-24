# WhisperX API — run targets in .venv
SHELL := /bin/bash
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
UVICORN := $(VENV_BIN)/uvicorn

DOCKER_REPOSITORY ?= kperreau/whisperx

.PHONY: venv deps deps-lint build run lint image push help

help:
	@echo "Targets:"
	@echo "  venv      - Create .venv"
	@echo "  deps      - Install/update deps (torch CPU + requirements); creates venv if needed"
	@echo "  deps-lint - Install ruff (for lint); creates venv if needed"
	@echo "  build     - Check that app imports (requires deps)"
	@echo "  run       - Run uvicorn (requires deps)"
	@echo "  lint      - Run ruff on the code (requires deps-lint)"
	@echo "  image     - Build Docker image (multi-arch, no push)"
	@echo "  push      - Build and push Docker image to $(DOCKER_REPOSITORY)"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "venv ready: $(VENV)"

deps: venv
	@$(PIP) install -q --upgrade pip
	@$(PIP) install -q --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
	@$(PIP) install -q --no-cache-dir -r requirements.txt
	@echo "deps ready"

deps-lint: venv
	@$(PIP) install -q ruff
	@echo "lint deps ready"

lint: deps-lint
	@$(VENV_BIN)/ruff check .

build: deps
	@$(PYTHON) -c "from app.main import app; print('ok')"

run: deps
	@$(UVICORN) app.main:app --host 0.0.0.0 --port 8000

image:
	@PUSH=false DOCKER_REPOSITORY="$(DOCKER_REPOSITORY)" ./scripts/docker-build-push.sh

push:
	@PUSH=true DOCKER_REPOSITORY="$(DOCKER_REPOSITORY)" ./scripts/docker-build-push.sh
