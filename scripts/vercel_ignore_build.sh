#!/usr/bin/env bash
# Vercel's ignored build step for the grader API (vercel.json ignoreCommand). Exit 0 skips the
# build and exit 1 builds; Vercel fails the deployment on any other status, so every other outcome
# is mapped to 1. The image holds only the API, the packages it imports, the locks its bake step
# reads and the held-out questions it grades, so a push that touches none of them (results,
# documents, requests, the site) builds nothing. Every such push used to build and push a 568 MB
# image, and the project's image repository reached its Hobby limit of 50 images on 1 November 2026.
#
# The comparison is with the last deployed commit when Vercel names it, so changes spread over
# several pushes still build. Vercel clones shallow, and that commit can be older than the clone
# reaches ("fatal: bad object" with status 128 on 1 November 2026), so it is fetched alone first;
# a diff needs only the two trees, not the history between them. Anything git cannot answer builds,
# which is the safe side. A redeploy from the dashboard can bypass this step with the box "Use
# project's Ignore Build Step" cleared, which is how a change to the environment variables reaches
# the running API without a commit.
set -u
base="${VERCEL_GIT_PREVIOUS_SHA:-HEAD^}"
if ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
  git fetch --quiet --depth=1 origin "$base" 2>/dev/null || exit 1
fi
if git diff --quiet "$base" HEAD -- \
  Dockerfile.vercel pyproject.toml uv.lock \
  packages/api packages/core packages/rewards packages/heldout packages/monitor packages/questions \
  data/sources.lock.json data/twins.lock.json \
  data/questions/readback_test.jsonl data/questions/test.jsonl data/questions/template_test.jsonl; then
  exit 0
fi
exit 1
