#!/usr/bin/env bash
# Commit and push .semver and _deploy/helm/values.yaml after a successful image build+push.
# Uses the same commit message as the triggering commit, with [skip ci] to avoid loops.

set -e

COMMIT_SHA=${COMMIT_SHA:-HEAD}
COMMIT_MESSAGE=$(git show -s --format=%B "$COMMIT_SHA")
COMMIT_USER_EMAIL=$(git show -s --format='%ae' "$COMMIT_SHA")
COMMIT_USER_NAME=$(git show -s --format='%an' "$COMMIT_SHA")

if [ -z "$(git config user.email 2>/dev/null)" ]; then
  git config --global user.email "$COMMIT_USER_EMAIL"
  git config --global user.name "$COMMIT_USER_NAME"
fi

git add .semver _deploy/helm/values.yaml

if git diff --cached --quiet; then
  echo "No version/config changes to commit."
  exit 0
fi

git commit -m "${COMMIT_MESSAGE}

https://github.com/${GITHUB_REPOSITORY:-kperreau/whisperx}/commit/${COMMIT_SHA}
update app version and helm values [skip ci]
by $COMMIT_USER_NAME $COMMIT_USER_EMAIL" || exit 0

git stash --include-untracked --quiet 2>/dev/null || true
git pull --rebase

git push

echo "✅ Release push completed successfully"
