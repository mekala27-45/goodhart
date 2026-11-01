---
name: request-a-gpu-run
description: Write, commit and promote a GPU run in goodhart. Use when a stage (probe, select, sft, pairs, dpo, grpo, sample, judge) must run on a Kaggle GPU, or when a results pull request from a gpu/<id> branch needs review and promotion.
---

# Request a GPU run

Nothing on this machine starts a GPU. A run happens only when a request file is committed under
`requests/` and the gpu workflow picks it up; its outputs come back on a branch for review.

## 1. Write the request

Create `requests/<local date>-<nn>-<name>.json`; the file name without `.json` is the `id`.

- `stage`: one of probe, select, sft, pairs, dpo, grpo, sample, judge.
- `model` (or `models` for probe and select), `reward` and `arm` for GRPO, `seed`.
- `hyperparameters`: only what differs from `config/training.yaml`; every key is read by the stage.
- `step_budget` and `gpu_hour_ceiling`: the ceiling counts every phase; a stage stops at it and
  says so in the ledger row.
- `inputs`: earlier request ids whose kernel output this run reads (a checkpoint to resume, an
  adapter to start from).
- `accelerator`: leave empty to use the probe's choice in `results/gpu_probe.json`.

## 2. Declare the data

```
uv run python scripts/new_request.py requests/<id>.json
```

This writes the sha256 of every data file the stage reads into `data`. The kernel builder refuses
a request whose digests differ from the files at its commit, so rebuild the data first if it
changed, then fill the request.

## 3. Check it builds

```
uv run python scripts/build_kernel.py --request requests/<id>.json --dry-run
```

The bundle never holds goodhart_heldout, the twins, the audit key or the gold of a held-out set;
the builder raises if it would.

## 4. Commit and push

One request per commit, with a message that says what the run decides. The gpu workflow runs every
request with no record (no `results/runs/<id>/provenance.json` on main and no `gpu/<id>` branch).

## 5. Review and promote

The workflow opens a pull request from `gpu/<id>`. Before merging:

- `python scripts/promote_results.py --check --run <id>` passes: every file matches the digest the
  kernel reported and the request is the one committed;
- the ledger row (`results/runs/<id>/ledger.jsonl`) shows the status, the steps, the GPU seconds by
  phase and whether the run stopped at its ceiling;
- `monitor.jsonl` has one row per step for a GRPO run, and `alerts.json` is read if present (the
  runbook is `docs/runbooks/reward-hacking.md`).

Merging is the promotion. Then grade: `uv run goodhart grade`, and commit the grades.
