#!/usr/bin/env bash
# Vercel's ignored build step for the grader API (vercel.json ignoreCommand). Exit 0 skips the
# build, any other status builds. The image holds only the API, the packages it imports, the locks
# its bake step reads and the held-out questions it grades, so a push that touches none of them
# (results, documents, requests, the site) builds nothing. Every such push used to build and push a
# 568 MB image, and the project's image repository reached its Hobby limit of 50 images on 1
# November 2026. The comparison is with the last deployed commit when Vercel names it, so changes
# spread over several pushes still build; any git error builds, which is the safe side.
set -u
base="${VERCEL_GIT_PREVIOUS_SHA:-HEAD^}"
git diff --quiet "$base" HEAD -- \
  Dockerfile.vercel pyproject.toml uv.lock \
  packages/api packages/core packages/rewards packages/heldout packages/monitor packages/questions \
  data/sources.lock.json data/twins.lock.json \
  data/questions/readback_test.jsonl data/questions/test.jsonl data/questions/template_test.jsonl
