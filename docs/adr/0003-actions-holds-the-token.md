# ADR 0003: GitHub Actions is the only machine with the Kaggle token

Status: accepted, 31 October 2026.

## Context

GPU runs happen on a free Kaggle quota, which a stray call can spend and a leaked token can spend
for someone else.

## Decision

Every GPU run is a committed request file. The gpu workflow, the only place the token exists (a
repository secret), builds the kernel, pushes it, waits, downloads the output, checks every digest
and opens a pull request from a results branch. Merging the pull request is the promotion.

## Alternatives and why they lost

- **The build machine calls Kaggle directly.** It would hold the token in its environment, and any
  mistake or injected instruction there could start a run or leak the token.
- **Kaggle notebooks run by hand.** Nothing would tie a result to the code that made it, and the
  run would not be repeatable from a file.
- **Results written straight to main by the workflow.** An unreviewed result would reach the pages.

## Consequences

The route is slower (a push, a queue, a review) and every result carries its request, kernel,
commit and digests. Token and identifier scans run on every commit diff in CI and in a pre-push
hook, so a credential never reaches a pushed blob.
