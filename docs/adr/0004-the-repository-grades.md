# ADR 0004: The repository grades every completion; the GPU never does

Status: accepted, 31 October 2026.

## Context

A kernel installs packages and downloads models from the internet. Its outputs are claims about
what a model wrote, not grades.

## Decision

Kernels export completions with their request digest and checkpoint. packages/evaluation parses,
executes and compares each one on the original and on the kept twins, and CI regrades all of them
on every push (`goodhart grade --check`). The API uses the same grading code, and parity between
the API verdict and the repository verdict is a promotion gate.

## Alternatives and why they lost

- **Grade in the kernel and export the grades.** The twins and the held-out gold would have to
  travel to the GPU, which is exactly what the held-out rule forbids, and a grade would be only as
  trustworthy as the kernel's environment.
- **Grade only a sample of completions.** Intervals over a sample would hide the per item truth the
  pages show.

## Consequences

Grading costs CPU time in CI, and every number on every page can be traced to graded rows in
results/grades.
