# goodhart

Post-training small open models to write SQL for questions about three warehouses, with SFT, DPO and
GRPO on free GPUs, and grading every answer with checks the training reward never saw.

> goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.

## What it is

Three warehouses built earlier in this series (doorstep for housing, turnaround for flights, ledger
for a fictional bank) hold 163 marts and 70,386,241 rows. A generator writes questions about
them with gold SQL and gold answers, 2,400 for training and 377 for test among them. Small open
models are post-trained on a Kaggle T4 to answer in SQL, and every answer is graded in this
repository twice: on the warehouse the reward saw, and on perturbed twins of it that no training
stage can read. A query that is right for the right reason stays right on the twins; a query that
is right by accident does not. The headline is twin confirmed accuracy: the share of questions
answered correctly on the original and on every kept twin.

Inside it is a small study of reward hacking. Four defects are planted in the reward on purpose (a
comparator that pays any table containing the answer, sample rows that leak it into the prompt, a
reward that pays any query that runs, a bonus for refusing), each with a prediction of what the
model will learn, which monitor will catch it and when. The predictions were hashed before any run,
and every page prints each one beside what happened.

## Where it stands

4 of the 18 lines of the definition of done are done. RESULTS.md has the evidence
for each line; the lines still open are the GPU stages that have not run or not been promoted yet,
and the documents that report on them.

| Line | Status | What it covers |
|---:|---|---|
| 1 | done | Warehouses fetched by lock and profiled, ten defects as rules, provenance |
| 2 | done | Comparator, rewards, twins and distillation, mutation score, detectors, tested |
| 3 | not done | Generator, splits, paraphrase round trip, held-out sets, contamination, planted defect |
| 4 | done | Plan, predictions, thresholds and hypotheses hashed before the first test score |
| 5 | not done | GPU request route, results branch, reviewed promotion, resume, ledger, no token |
| 6 | not done | Probe and base selection by coverage with licenses; base, floor and reference graded |
| 7 | not done | SFT clean and planted with the audit; DPO with its pair counts |
| 8 | not done | Grader API live with the grade log, verified from a separate client; parity exact |
| 9 | not done | GRPO M3 and M4 with the honest reward and the learnability filter, seeds, ablation |
| 10 | not done | Four defect arms and two controls, predictions held or not, the incident |
| 11 | not done | Twin confirmed accuracy, pass at k, intervals, BH, interior tests, hypotheses, regression |
| 12 | not done | Judge calibration: agreement, kappa, false accepts per hack type |
| 13 | not done | Persistence checks and latency with the cold start |
| 14 | not done | Sizing sheet and compute ledger, every run inside its ceiling |
| 15 | not done | Model cards, validation reports, promotion gates, champion, adapters released |
| 16 | not done | Site live on Pages with every route; the desk grading live; the live check |
| 17 | done | RESULTS, README with the matrix, DECISIONS, notebooks, AGENTS.md, skills, SPEC, ADRs |
| 18 | not done | Coverage, types, lint, gates, forty commits, rederive, v0.1.0, licenses, repo described |

The site is at https://mekala27-45.github.io/goodhart/ and the grader API at
https://goodhart-grader.vercel.app (POST /v1/grade with a question id and a query).

## The skills matrix

| Skill | Where it lives |
|---|---|
| Supervised fine-tuning with LoRA | packages/train/src/goodhart_train/sft.py, the SFT model cards |
| Preference optimization (DPO) | packages/train/src/goodhart_train/dpo.py, the pair counts |
| RL with verifiable rewards (GRPO) | packages/train/src/goodhart_train/grpo.py, packages/rewards |
| Algorithm selection | the methods page, the recommendation rules |
| Reward design and reward hacking | packages/rewards, packages/monitor, the defect arms, docs/incidents |
| Verifier design and strength | packages/heldout, results/mutation.json |
| Experiment design and ablations | evaluation/plan.yaml, the controls, the ablation, the seeds |
| Evaluation and held-out testing | packages/evaluation, the held-out sets |
| LLM as judge, calibrated | packages/judge, the verifier page |
| Data strategy and synthetic data | packages/questions, the paraphrase round trip, the filters |
| Data quality diagnosis | the planted defect and the audit (packages/evaluation/src/goodhart_evaluation/audit.py) |
| Contamination control | results/contamination.json |
| Base model selection | results/selection.json, the models page |
| Workload sizing and cost | packages/sizing, the sizing page |
| GPU orchestration on free hardware | .github/workflows/gpu.yml, kernels/, packages/train |
| Statistics for evaluation | pass at k, the cluster bootstrap, Benjamini-Hochberg, the interior tests |
| SQL and DuckDB | the warehouses, packages/rewards/src/goodhart_rewards/compare.py |
| Serving and persistence | packages/api, the grade log on Neon, the parity check |
| Model documentation and release | docs/models, docs/validation, the release assets |
| Agent assisted engineering | AGENTS.md, .claude/skills, docs/SPEC.md |
| Communication | report/briefing.md, the objection at the end of every page |

## How it runs

Everything that touches a GPU starts from a file. A request under requests/ names a stage, a model,
a reward, a seed, the hyperparameters, a ceiling in GPU hours and the digest of every data file the
stage reads. Pushing it starts the gpu workflow, the only machine that holds the Kaggle token: it
builds a bundle that carries no held-out gold and no twin, runs it on Kaggle, checks the digest of
every output and opens a pull request with the results and their provenance. Merging that pull
request is the promotion. The kernel writes completions and never a grade; this repository parses,
executes and compares every completion again, on the original and on the twins, with the code the
API uses.

```
uv sync --frozen
make fetch       # the warehouses and readback, checked by digest
make data        # questions, twins, mutation score, contamination, prompts, SFT targets
make gates       # dashes, vocabulary, statement, tokens, identifiers, registration, skills, palette
make test        # the whole suite, the trainers on CPU with a tiny model
make manifest    # every published number, from the files the steps wrote
make render      # README, RESULTS and the reports, from the manifest
```

ARCHITECTURE.md explains the design and why each part is there, DECISIONS.md records every choice
with its date, AGENTS.md is the entry point for a coding agent, and docs/SPEC.md is the brief the
build follows.

## Licenses

The code is Apache 2.0. The warehouses keep the terms their own repositories state, restated in
data/PROVENANCE.md with every other source: readback's catalog and golden set, spaCy's English
vectors (MIT), Spider (CC BY-SA 4.0) and IFEval (Apache 2.0). Each base model keeps the license its
card declares, recorded on the day in results/gpu_probe.json; an adapter is released only for a base
whose license permits it.
