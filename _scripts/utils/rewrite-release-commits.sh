#!/usr/bin/env bash
# One-off: rewrite release commits to remove personal email from author and from message.
# Run from repo root, then: git push --force-with-lease origin main
# Safe to delete this script after use.

set -e

BRANCH="${1:-main}"
BOT_NAME="github-actions[bot]"
BOT_EMAIL="github-actions[bot]@users.noreply.github.com"

echo "Rewriting release commits on branch: $BRANCH"
echo "  - Removing 'by Name <email>' from commit messages"
echo "  - Setting author/committer to $BOT_NAME <$BOT_EMAIL> for those commits"
echo ""

git filter-branch -f \
  --msg-filter 'sed "/^by .*$/d"' \
  --env-filter "
    MSG=\$(git log -1 --format=%B \$GIT_COMMIT)
    if echo \"\$MSG\" | grep -q 'update app version and helm values \[skip ci\]'; then
      export GIT_AUTHOR_NAME='$BOT_NAME'
      export GIT_AUTHOR_EMAIL='$BOT_EMAIL'
      export GIT_COMMITTER_NAME='$BOT_NAME'
      export GIT_COMMITTER_EMAIL='$BOT_EMAIL'
    fi
  " \
  -- "$BRANCH"

echo ""
echo "Done. To update the remote (rewrites history):"
echo "  git push --force-with-lease origin $BRANCH"
echo ""
echo "If others have cloned the repo, they should re-clone or rebase after you push."
