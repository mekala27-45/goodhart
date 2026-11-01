# Results

> goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.

Every number below is read from results/manifest.json, which the build assembles from the files
each step wrote (manifest as of 2026-11-01). A section whose step has not run says so
instead of printing a number. Sections not in the manifest yet: judge, release, interior, checkpoints.

## The definition of done

7 of 18 lines are done. Each line carries one sentence of evidence from the files the
build wrote; none is marked by hand.

| Line | Status | What it covers | Evidence |
|---:|---|---|---|
| 1 | done | Warehouses fetched by lock and profiled, ten defects as rules, provenance | results/profile.json profiles 163 marts and 70,386,241 rows at full size with 10 defects written as rules; data/PROVENANCE.md and data/sources.lock.json record every source by digest. |
| 2 | done | Comparator, rewards, twins and distillation, mutation score, detectors, tested | The kept twins (twin-13, twin-01, twin-12, twin-07) confirm 16 of 2,482 non equivalent mutants of the test golds, of which the honest reward pays 112; tests/verifier and tests/heldout test the comparator, every reward, the twins and the detectors. |
| 3 | done | Generator, splits, paraphrase round trip, held-out sets, contamination, planted defect | data/questions holds train 2,416, dev 240, test 377, template_test 156, readback_test 84, readback_dev 40; results/contamination.json moved 27 held-out items; 120 SFT targets carry the planted defect, its key hashed in data/audit_key.sha256; the teacher's rewrites of 600 training questions were graded and 16 kept (data/paraphrases/kept.jsonl). |
| 4 | done | Plan, predictions, thresholds and hypotheses hashed before the first test score | evaluation/registration.json registered the plan (f4b75fe30811), the predictions, the monitor thresholds and the hypotheses on 2026-10-31, before the first test question was scored; scripts/check_registration.py holds every later change to a dated decision. |
| 5 | done | GPU request route, results branch, reviewed promotion, resume, ledger, no token | 8 runs came back on gpu/<id> branches with provenance and were promoted by pull request; resume and the ledger are tested in tests/train/test_smoke.py; the token scan reads every file and every commit diff. |
| 6 | done | Probe and base selection by coverage with licenses; base, floor and reference graded | results/gpu_probe.json probed 8 candidates; results/selection.json chose Qwen/Qwen2.5-Coder-1.5B-Instruct and Qwen/Qwen3-0.6B by the registered rule; the floor is graded. |
| 7 | not done | SFT clean and planted with the audit; DPO with its pair counts | No promoted sft or dpo run yet. The audit flagged 95 targets, 95 of them planted. |
| 8 | not done | Grader API live with the grade log, verified from a separate client; parity exact | results/parity.json or results/live_check.json is missing: the live API is not verified yet. |
| 9 | not done | GRPO M3 and M4 with the honest reward and the learnability filter, seeds, ablation | No promoted GRPO run yet for M3, M4. |
| 10 | not done | Four defect arms and two controls, predictions held or not, the incident | The defect study has no promoted run yet for honest, lenient, leaked, lazy, refusal, random, format. |
| 11 | not done | Twin confirmed accuracy, pass at k, intervals, BH, interior tests, hypotheses, regression | Missing: held-out grades for every method (1 of five graded); every hypothesis tested; the interior tests; IFEval for every method. |
| 12 | not done | Judge calibration: agreement, kappa, false accepts per hack type | results/judge.json is missing: the judge has not answered its registry yet. |
| 13 | not done | Persistence checks and latency with the cold start | results/persistence.json or results/latency.json is missing. |
| 14 | not done | Sizing sheet and compute ledger, every run inside its ceiling | The ledger sums 3.366 GPU hours over the promoted runs. |
| 15 | not done | Model cards, validation reports, promotion gates, champion, adapters released | The model cards, the validation reports or the release record is missing. |
| 16 | not done | Site live on Pages with every route; the desk grading live; the live check | results/live_check.json is missing: the site and the desk are not checked live yet. |
| 17 | done | RESULTS, README with the matrix, DECISIONS, notebooks, AGENTS.md, skills, SPEC, ADRs | DECISIONS.md has 25 dated entries, 3 of them reversals; 3 notebooks; 7 ADRs; 3 skills; AGENTS.md and docs/SPEC.md are in place. |
| 18 | not done | Coverage, types, lint, gates, forty commits, rederive, v0.1.0, licenses, repo described | 148 commits on main when this manifest was built; the gates (dashes, vocabulary, palette, skills, tokens, identifiers, types, lint, coverage at 80 percent) run in CI on every push; missing: the rederive; the v0.1.0 tag. |

## The questions

The generator drew 10,664 candidate questions from the templates and the catalog, kept a pool of
5,731 after the filters, and split them by entity tuple and template. readback's own golden set gives
the readback test and development sets.

| Set | Questions |
|---|---:|
| train | 2,416 |
| dev | 240 |
| test | 377 |
| template_test | 156 |
| readback_test | 84 |
| readback_dev | 40 |

By family:

| set | lookup | comparison | trend | top_k | multi | refusal |
|---|---:|---:|---:|---:|---:|---:|
| train | 485 | 433 | 360 | 338 | 553 | 247 |
| dev | 48 | 43 | 36 | 34 | 55 | 24 |
| test | 80 | 72 | 60 | 49 | 76 | 40 |
| template_test | 0 | 0 | 0 | 0 | 156 | 0 |
| readback_test | 15 | 17 | 15 | 17 | 10 | 10 |
| readback_dev | 9 | 7 | 9 | 7 | 3 | 5 |

What the filters dropped, by rule:

| reason | dropped |
|---|---:|
| contamination moved out of template_test | 4 |
| contamination moved out of test | 23 |
| duplicate SQL | 1,778 |
| duplicate question | 451 |
| held-out shape outside the test buckets | 2,614 |
| readback golden tuple | 90 |
| twins cannot check | 31 |

120 SFT targets carry the planted data defect, its key hashed in data/audit_key.sha256 before the
audit ran. The leaked defect arm shows sample rows that include the answer row on 931 training
prompts (38.5%), the share where such a row exists at the asked grain.

## The verifier

The twins keep every key, label and year of a table and move every measure by a seeded factor. Of
sixteen candidates, greedy cover over the train and development mutants kept 4:
twin-13, twin-01, twin-12, twin-07. They catch 139 of the 167 distillation mutants the original
warehouse pays.

On mutants of the test golds, which the distillation never saw: of 3,170 mutants,
688 return the gold result on every version and are set aside as equivalent. Of the
2,482 that remain, the honest comparator pays 112 (4.5%), the lenient comparator
of the first defect pays 973 (39.2%), and the original with every kept twin confirms
16 (0.6%).

| family | mutants | equivalent | scored | honest pays | lenient pays | twins confirm |
|---|---:|---:|---:|---:|---:|---:|
| comparison | 760 | 217 | 543 | 1 | 230 | 0 |
| lookup | 495 | 77 | 418 | 6 | 6 | 0 |
| multi | 822 | 117 | 705 | 63 | 248 | 6 |
| top_k | 545 | 162 | 383 | 42 | 189 | 10 |
| trend | 548 | 115 | 433 | 0 | 300 | 0 |
| all | 3,170 | 688 | 2,482 | 112 | 973 | 16 |

## Contamination

Every held-out item was checked against training and development by exact text, entity tuple,
8-gram overlap, embedding cosine at 0.95 and character n-gram cosine. A near copy moves out
of a held-out set only when it asks about the same subject and has the same gold answer.

| Check | Count |
|---|---:|
| exact duplicates | 0 |
| shared entity tuples | 0 |
| 8-gram overlaps | 2 |
| embedding near duplicates | 27 |
| character n-gram near duplicates | 6 |
| held-out items moved out | 27 |
| template neighbours kept (a new entity or year, not a copy) | 19 |
| embedding neighbours kept (another answer) | 576 |

## The first look

The profile read 163 marts and 70,386,241 rows at full size and wrote the first ten defects it
found as rules the generator applies.

| defect | warehouse | count | rule |
|---|---|---:|---|
| turnaround_partial_2026 | turnaround | 5 | annual turnaround questions never ask about 2026; the generator excludes it |
| turnaround_short_carrier_years | turnaround | 17 | a carrier year is asked about only when it has all twelve months |
| ledger_edge_quarters | ledger | 2 | ledger questions use the full quarters 2023Q4 to 2026Q2 only, as readback's floor does |
| doorstep_payment_share_nulls | doorstep | 18,039 | a question is drawn only from cells where its measure is not null; a gold with a null drops |
| doorstep_years_after_2024 | doorstep | 1,870 | a period with no value is out of range, which is a refusal item, never a lookup |
| doorstep_two_kinds | doorstep | 14,634 | metro measures carry the catalog's kind filter, and the gold keeps it as a predicate |
| doorstep_quarantine | doorstep | 593 | no question is drawn from a quarantine mart; a schema block may still list one, as a real table |
| turnaround_quarantine | turnaround | 29 | no question is drawn from a quarantine mart; a schema block may still list one, as a real table |
| turnaround_airports_beyond_catalog | turnaround | 396 | airport questions name only airports in the catalog's entity list |
| doorstep_rate_as_percent | doorstep | 19 | units come from the catalog per measure; a percent never compares with a fraction |

## The GPU

The kernels ran on NvidiaTeslaT4 with vllm generation.

| model | license | parameters | passes | tokens per second | why not |
|---|---|---:|---|---:|---|
| HuggingFaceTB/SmolLM3-3B | apache-2.0 | 3,075,098,624 | no | 66.7 | OutOfMemoryError: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 36.8 |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | apache-2.0 | 1,543,714,304 | yes | 102.0 |  |
| Qwen/Qwen3-0.6B | apache-2.0 | 596,049,920 | yes | 82.1 |  |
| Qwen/Qwen3-1.7B | apache-2.0 | 1,720,574,976 | yes | 66.1 |  |
| Qwen/Qwen3.5-0.8B | apache-2.0 | 752,393,024 | yes | 213.9 |  |
| Qwen/Qwen3.5-2B | apache-2.0 | 1,881,825,088 | yes | 129.7 |  |
| google/gemma-4-E2B-it |  | 0 | no | 0.0 | Loading weights:  10%|▉         | 189/1951 [00:00<00:07, 247.44it/s]
Loading weights:  13%|█▎        | 246/1951 [00:00<0 |
| meta-llama/Llama-3.2-1B-Instruct | llama3.2 | 0 | no | 0.0 | OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llam |

## The floor

readback's no model floor, its grammar over its own golden families run through its own planner at
the pinned commit, is graded like any model: twin confirmed, on every held-out set. It is the line a
trained model has to clear, and it is strongest on the questions it was written for.

| set | items | twin confirmed | answerable refused | planner raised |
|---|---:|---:|---:|---:|
| readback_test | 84 | 91.9% | 3 | 0 |
| test | 377 | 35.3% | 112 | 1 |
| template_test | 156 | 0.0% | 35 | 0 |
| dev | 240 | 31.0% | 92 | 3 |
| readback_dev | 40 | 91.4% | 0 | 0 |

## The methods on one budget

Each method may spend 2.0 h of GPU time, counting every stage it needs, as the run ledger measured it. On
readback_test, the headline set, the table gives greedy twin confirmed accuracy with its interval,
execution accuracy on the original beside it, and pass at one and at sixteen from the sampled answers
at the registered temperature. The floor is readback's own planner, strongest on the questions it was
written for.

| method | twin confirmed | low | high | execution | pass at 1 | pass at 16 | gpu hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 10.8% | 4.6% | 18.3% | 16.2% | 6.5% | 32.4% | 0.00 h |
| floor | 91.9% | 80.0% | 100.0% | 91.9% | not available | not available | not available |

- The base trailed the floor on twin confirmed accuracy by 81.1 points (95 percent interval -94.1 to -67.0 points).

Every set, every metric and every interval is in the methods table of the site and in
results/methods.json.

## The hypotheses

Each hypothesis was registered before the test sets were scored and is tested once, from the
repository's own grades, with its verdict printed whatever it is. The p values are one sided in the
direction each one predicts, adjusted by Benjamini-Hochberg across the hypotheses that could be
tested; one whose inputs were never produced says which.

| hypothesis | statement | estimate | low | high | adjusted p | verdict | note |
|---|---|---:|---:|---:|---:|---|---|
| H1 | SFT then GRPO beats SFT alone on twin confirmed accuracy on readback_test | not available | not available | not available | not available | not measured | not produced: M4, M1 |
| H2 | GRPO raises pass at 1 more than pass at 16 | not available | not available | not available | not available | not measured | not produced: M31, M316 |
| H3 | the random reward reaches less than a third of the honest gain | not available | not available | not available | not available | not measured | not produced: start, honest, random |
| H4 | the lenient arm opens a twin gap of at least ten points by its last checkpoint | not available | not available | not available | not available | not measured | not produced: honest, lenient |
| H5 | DPO on verified self samples reaches half the GRPO gain at a quarter of the compute | not available | not available | not available | not available | not measured | not produced: M2, M3, M2 GPU hours, M3 GPU hours |

## The regression suite

Training on three warehouses can cost a model what it could do before. IFEval measures whether it
still follows plain instructions, graded by Google's reference checker; each change is against the
base, paired by prompt, with its interval, and the verdict word comes from the interval alone.

| method | accuracy | low | high | change | change low | change high | verdict | items |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| M0 | 37.5% | 33.5% | 41.4% | not available | not available | not available | the base | 541 |

Spider dev measures text to SQL on databases no training item touches, under the test suite
evaluation: an answer is right only when it matches the gold on the original database and on every
test suite database of its schema. It is a public benchmark, likely in every candidate's
pretraining data, so it reads as a transfer check in both directions and never as skill.

| method | accuracy | low | high | change | change low | change high | verdict | items |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| M0 | 52.2% | 43.6% | 61.3% | not available | not available | not available | the base | 1,034 |

## The audit

The audit re-executed every SFT target against its declared answer before the key was opened. It
flagged 95 targets: 95 of the 120 planted, 0 false positives, 25 missed.

## Limitations

- The questions are generated from templates over three warehouses. A model that does well here
  has learned this catalog and these shapes; the template test set measures how far that carries
  to shapes no training template has, and nothing here measures questions people actually ask.
- The twins are perturbations of the same tables, not new data. They catch a query that is right
  by accident on these values; they cannot catch a query that is wrong in a way the perturbation
  leaves alone, such as a filter on a key that every twin keeps.
- The comparator decides what counts as the same answer: one label column beside a value is
  allowed, a percent never matches a fraction, extra rows never pass. Each rule is a choice, tested
  and written down, and a different choice would move every number.
- The cards are T4s: float16 only, no FlashAttention, and a session quota counted in hours. The
  cheaper methods get more steps under one ceiling than a larger card would give them, and the
  sizing page projects to larger models as estimates, not measurements.
- The defect arms run on the small model only, with one seed each where the quota allows no more.
  A hack the small model finds need not be found by the main model, or at the same step.
- The judge is one model from one family, calibrated against the twins on a registry of graded
  completions. It is never used inside a reward.
- Spider is very likely in the pretraining data of every candidate, so it is reported as a
  regression check and never as evidence of skill.
