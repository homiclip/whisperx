#!/usr/bin/env bash
# Build and optionally push a multi-arch Docker image (linux/amd64, linux/arm64) with docker buildx.
# linux/arm64 is used for Apple Silicon (M1/M2/M3). Building arm64 on an amd64 host (Intel Mac, CI)
# requires QEMU; the GitHub workflow uses docker/setup-qemu-action. For local Linux:
#   docker run --privileged --rm tonistiigi/binfmt --install all
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

buildCmd=(docker buildx build --platform="linux/amd64,linux/arm64" --network host)

if [[ "${PUSH}" == "true" ]]; then
  buildCmd+=(--push)
fi

echo "Executing Docker command:"
echo "${buildCmd[*]}" -t "${DOCKER_REPOSITORY}:latest" -t "${DOCKER_REPOSITORY}:${GIT_VERSION}" -f "${DOCKERFILE}" .
"${buildCmd[@]}" -t "${DOCKER_REPOSITORY}:latest" -t "${DOCKER_REPOSITORY}:${GIT_VERSION}" -f "${DOCKERFILE}" .
