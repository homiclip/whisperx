#!/usr/bin/env bash
# Commit and push .semver and _deploy/helm/values.yaml after a successful image build+push.
# Uses the same commit message as the triggering commit, with [skip ci] to avoid loops.
# Author/committer are always the CI bot (no personal email in history).

set -e

COMMIT_SHA=${COMMIT_SHA:-HEAD}
COMMIT_MESSAGE=$(git show -s --format=%B "$COMMIT_SHA")

# Use neutral bot identity so release commits never expose personal emails
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-github-actions[bot]}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-github-actions[bot]@users.noreply.github.com}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"

git add .semver _deploy/helm/values.yaml

if git diff --cached --quiet; then
  echo "No version/config changes to commit."
  exit 0
fi

git commit -m "${COMMIT_MESSAGE}

https://github.com/${GITHUB_REPOSITORY:-kperreau/whisperx}/commit/${COMMIT_SHA}
update app version and helm values [skip ci]" || exit 0

git stash --include-untracked --quiet 2>/dev/null || true
git pull --rebase

git push

echo "✅ Release push completed successfully"
