# ADR 0006: The reward executes SQL on DuckDB inside the training process

Status: accepted, 31 October 2026.

## Context

GRPO pays 64 completions per optimizer step; the reward must execute each one, once, fast.

## Decision

The kernel fetches the original marts by lock and opens them read only in DuckDB with the same
guard as everywhere else (one read only SELECT over warehouse tables, no file or network access,
a row cap and a timeout). The reward runs in the trainer's process.

## Alternatives and why they lost

- **A remote SQL sandbox service.** A network call per completion, a second service to secure and
  keep awake, and timeouts that read as wrong answers; and the reward would depend on a service
  the run cannot pin by digest.
- **Executing on the twins during training.** It would let the reward see the held-out check,
  which defeats it.

## Consequences

The reward sees exactly the original warehouse and nothing else. DuckDB runs on one thread so a
floating point sum comes back the same on every run.
