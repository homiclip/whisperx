#!/usr/bin/env bash
# Build and optionally push a Docker image with docker buildx.
# Platform: linux/amd64 only. linux/arm64 is not supported because pyannote-audio
# (whisperx dependency) requires torchcodec, which has no official wheels for arm64.
# On Apple Silicon, the amd64 image runs via emulation.
#
# Run from the project root. Set PUSH=true to push to the registry.
#
#   PUSH=false              ./scripts/docker-build-push.sh   # build only
#   PUSH=true               ./scripts/docker-build-push.sh   # build and push
#   DOCKER_REPOSITORY=user/repo  ./scripts/docker-build-push.sh

set -e

: "${PUSH:="false"}"
: "${DOCKER_REPOSITORY:="kperreau/whisperx"}"
: "${PROJECT_PATH:="."}"

ROOT="$(cd "${PROJECT_PATH}" && pwd)"
cd "${ROOT}"
DOCKERFILE="Dockerfile"

GIT_VERSION="$(git rev-parse --short=7 HEAD 2>/dev/null || echo "norev")"

buildCmd=(docker buildx build --platform="linux/amd64" --network host)

if [[ "${PUSH}" == "true" ]]; then
  buildCmd+=(--push)
fi

echo "Executing Docker command:"
echo "${buildCmd[*]}" -t "${DOCKER_REPOSITORY}:latest" -t "${DOCKER_REPOSITORY}:${GIT_VERSION}" -f "${DOCKERFILE}" .
"${buildCmd[@]}" -t "${DOCKER_REPOSITORY}:latest" -t "${DOCKER_REPOSITORY}:${GIT_VERSION}" -f "${DOCKERFILE}" .
