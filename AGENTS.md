# AGENTS.md

The single source for anyone, person or agent, working in this repository. CLAUDE.md and
.github/copilot-instructions.md point here.

goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.

## Setup

```
uv sync --frozen            # Python 3.12, the workspace and the dev tools
make hooks                  # the pre-push hook: token and identifier scans on every pushed commit
make fetch                  # the warehouses and the readback files, by sha256, into data/cache
uv run goodhart twins       # the four kept twins, rebuilt from data/twins.lock.json, every file checked
```

The trainers need the smoke extra on CPU (`uv pip install` CPU PyTorch, then the pinned TRL stack,
as in .github/workflows/ci.yml). No GPU is used from this machine, ever: GPU runs are requested by
committed files and run by GitHub Actions on Kaggle.

## Tests and gates

```
make gates                  # em dash, vocabulary, statement, token, identifier, registration, palette
make lint types             # ruff (120 columns) and mypy strict
uv run pytest               # the suite; named markers skip what is missing, CI requires all of them
uv run goodhart rebuild-check   # the questions, twins and mutants rebuild to the committed bytes
uv run goodhart grade --check   # every exported completion graded again
python scripts/promote_results.py --check   # every promoted GPU run matches its provenance
```

CI runs all of it with `GOODHART_REQUIRE=postgres,warehouses,torch`, so a skip there is a failure.
Coverage counts every module of every package and must stay at or above 80 percent.

## How work flows

- Questions, twins and the mutation score: `goodhart data` (about ten minutes), then commit the
  outputs it lists. The build is deterministic; the rebuild check proves it under another hash seed.
- A GPU run: write `requests/<date>-<nn>-<name>.json`, run `python scripts/new_request.py` on it to
  declare the digest of every data file the stage reads, commit, push. The gpu workflow runs it and
  opens a pull request from `gpu/<id>`. Review the digests and the summary; merging is the
  promotion. See `.claude/skills/request-a-gpu-run`.
- Grading: `goodhart grade` reads every promoted completions file; nothing is graded on the GPU.
- Numbers in prose: only through the claim gate. A number in README, RESULTS or any page comes from
  `results/manifest.json` through a template; `scripts/check_published_numbers.py` fails on a
  hand typed one.

## Files an agent must not edit

- The lock files: `data/sources.lock.json`, `data/twins.lock.json`, `uv.lock`,
  `requirements-gpu.txt` (they change only by the commands that write them).
- The registered plan: `evaluation/plan.yaml`, `evaluation/predictions.yaml`,
  `evaluation/hypotheses.yaml`, `config/monitor.yaml`, and the registration record
  `evaluation/registration.json`. A change after registration needs a dated DECISIONS.md entry that
  names the new hash, and `scripts/check_registration.py` enforces it.
- The audit key: `data/audit_key.sha256` before the audit, `data/audit_key.json` after it.
- Anything under `results/runs/` except through the promotion of a results pull request.
- Pushed history: never rewritten, never force pushed.

## Rules that are not negotiable

- No credential in any file or commit. The scans run in the pre-push hook and in CI.
- No test question reaches a training, filtering, prompt design, reward design or checkpoint
  choice. The kernel bundle refuses test gold and the held-out package; SFT, DPO and GRPO refuse
  any held-out id.
- No reward without its three tests, no run without its request, no metric without its interval.
- Every surface carries the statement above.
- Plain prose: no em dashes, none of the banned phrases in `scripts/check_vocabulary.py`.
