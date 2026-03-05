#!/usr/bin/env bash
# Build and optionally push a Docker image with docker buildx.
# Uses .semver for versioning: reads from origin/main or local .semver, bumps patch,
# tags image as <repo>:whisperx-vX.Y.Z and <repo>:latest, then updates .semver and
# _deploy/helm/values.yaml when PUSH=true.
#
# Platform: linux/amd64 only (pyannote-audio/torchcodec no arm64 wheels).
#
# Run from the project root:
#   PUSH=false                    ./_scripts/docker-build-push.sh   # build only
#   PUSH=true                     ./_scripts/docker-build-push.sh   # build and push
#   DOCKER_REPOSITORY=user/repo   ./_scripts/docker-build-push.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/semver.sh"
source "${SCRIPT_DIR}/utils/update_version.sh"

: "${PUSH:="false"}"
: "${DOCKER_REPOSITORY:="kperreau/whisperx"}"
: "${PROJECT_PATH:="."}"

ROOT="$(cd "${PROJECT_PATH}" && pwd)"
cd "${ROOT}"
DOCKERFILE="Dockerfile"

# --- Get current version from origin/main or local .semver ---
if git cat-file -e "origin/main:.semver" 2>/dev/null; then
  current_version=$(git show "origin/main:.semver")
else
  current_version=$([ -f .semver ] && cat .semver || echo "0.0.1")
fi

# --- Bump version ---
VERSION=$(bump_patch "${current_version}")
VERSION_TAG="v${VERSION}"

echo "Building Docker image tags..."
echo "  - ${DOCKER_REPOSITORY}:${VERSION_TAG}"
echo "  - ${DOCKER_REPOSITORY}:latest"
[[ "${PUSH}" == "true" ]] && echo "...and pushing them!"

buildCmd=(docker buildx build --platform="linux/amd64" --network host)

if [[ "${PUSH}" == "true" ]]; then
  buildCmd+=(--push)
fi

echo "Executing Docker command:"
echo "${buildCmd[*]}" -t "${DOCKER_REPOSITORY}:${VERSION_TAG}" -t "${DOCKER_REPOSITORY}:latest" -f "${DOCKERFILE}" .
"${buildCmd[@]}" -t "${DOCKER_REPOSITORY}:${VERSION_TAG}" -t "${DOCKER_REPOSITORY}:latest" -f "${DOCKERFILE}" .

# --- On success with PUSH, update .semver and values.yaml ---
if [[ $? -eq 0 ]] && [[ "${PUSH}" == "true" ]]; then
  update_image_tag_and_semver
fi
