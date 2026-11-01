#!/usr/bin/env bash
# Vercel's ignored build step for the grader API (vercel.json ignoreCommand). Exit 0 skips the
# build and exit 1 builds; Vercel fails the deployment on any other status, so every other outcome
# is mapped to 1. The image holds only the API, the packages it imports, the locks its bake step
# reads and the held-out questions it grades, so a push that touches none of them (results,
# documents, requests, the site) builds nothing. Every such push used to build and push a 568 MB
# image, and the project's image repository reached its Hobby limit of 50 images on 1 November 2026.
#
# The comparison is with the last deployed commit when Vercel names it, so changes spread over
# several pushes still build. Vercel clones shallow and without a usable origin, and that commit
# can be older than the clone reaches ("fatal: bad object" with status 128 on 1 November 2026), so
# it is fetched alone from the public repository first; a diff needs only the two trees, not the
# history between them. Anything git cannot answer builds, which is the safe side, and the step
# says in the build log what it compared and what it decided. A redeploy from the dashboard can
# bypass this step with the box "Use project's Ignore Build Step" cleared, which is how a change to
# the environment variables reaches the running API without a commit.
set -u
base="${VERCEL_GIT_PREVIOUS_SHA:-HEAD^}"
say() { echo "ignored build step: $*" >&2; }
if ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
  repo="https://github.com/${VERCEL_GIT_REPO_OWNER:-mekala27-45}/${VERCEL_GIT_REPO_SLUG:-goodhart}.git"
  if ! git fetch --quiet --depth=1 "$repo" "$base" 2>/dev/null; then
    say "the last deployed commit ${base} cannot be fetched, so this builds"
    exit 1
  fi
fi
if git diff --quiet "$base" HEAD -- \
  Dockerfile.vercel pyproject.toml uv.lock \
  packages/api packages/core packages/rewards packages/heldout packages/monitor packages/questions \
  data/sources.lock.json data/twins.lock.json \
  data/questions/readback_test.jsonl data/questions/test.jsonl data/questions/template_test.jsonl; then
  say "nothing the image holds changed since ${base}, so this skips"
  exit 0
fi
say "the image changes since ${base}, so this builds"
exit 1
