# Results

> goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.

Every number below is read from results/manifest.json, which the build assembles from the files
each step wrote (manifest as of 2026-11-01). A section whose step has not run says so
instead of printing a number. Sections not in the manifest yet: probe, paraphrases, regression, hypotheses, judge, release, methods, models, gates, interior, checkpoints, selection, audit, sizing.

## The definition of done

4 of 18 lines are done. Each line carries one sentence of evidence from the files the
build wrote; none is marked by hand.

| Line | Status | What it covers | Evidence |
|---:|---|---|---|
| 1 | done | Warehouses fetched by lock and profiled, ten defects as rules, provenance | results/profile.json profiles 163 marts and 70,386,241 rows at full size with 10 defects written as rules; data/PROVENANCE.md and data/sources.lock.json record every source by digest. |
| 2 | done | Comparator, rewards, twins and distillation, mutation score, detectors, tested | The kept twins (twin-13, twin-01, twin-12, twin-07) confirm 16 of 2,482 non equivalent mutants of the test golds, of which the honest reward pays 112; tests/verifier and tests/heldout test the comparator, every reward, the twins and the detectors. |
| 3 | not done | Generator, splits, paraphrase round trip, held-out sets, contamination, planted defect | data/questions holds train 2,400, dev 240, test 377, template_test 156, readback_test 84, readback_dev 40; results/contamination.json moved 27 held-out items; 120 SFT targets carry the planted defect, its key hashed in data/audit_key.sha256; the paraphrase round trip waits for the selection kernel. |
| 4 | done | Plan, predictions, thresholds and hypotheses hashed before the first test score | evaluation/registration.json registered the plan (f4b75fe30811), the predictions, the monitor thresholds and the hypotheses on 2026-10-31, before the first test question was scored; scripts/check_registration.py holds every later change to a dated decision. |
| 5 | not done | GPU request route, results branch, reviewed promotion, resume, ledger, no token | The route runs (two probes came back on results branches with provenance), and none is promoted to main yet. |
| 6 | not done | Probe and base selection by coverage with licenses; base, floor and reference graded | results/gpu_probe.json or results/selection.json is missing: the selection has not run. |
| 7 | not done | SFT clean and planted with the audit; DPO with its pair counts | No promoted sft or dpo run yet. The audit is not revealed yet. |
| 8 | not done | Grader API live with the grade log, verified from a separate client; parity exact | results/parity.json or results/live_check.json is missing: the live API is not verified yet. |
| 9 | not done | GRPO M3 and M4 with the honest reward and the learnability filter, seeds, ablation | No promoted GRPO run yet for M3, M4. |
| 10 | not done | Four defect arms and two controls, predictions held or not, the incident | The defect study has no promoted run yet for honest, lenient, leaked, lazy, refusal, random, format. |
| 11 | not done | Twin confirmed accuracy, pass at k, intervals, BH, interior tests, hypotheses, regression | Missing: held-out grades for every method (0 of five graded); every hypothesis tested; the interior tests; IFEval for every method; Spider for the base and the champion. |
| 12 | not done | Judge calibration: agreement, kappa, false accepts per hack type | results/judge.json is missing: the judge has not answered its registry yet. |
| 13 | not done | Persistence checks and latency with the cold start | results/persistence.json or results/latency.json is missing. |
| 14 | not done | Sizing sheet and compute ledger, every run inside its ceiling | results/sizing.json is missing: the sizing sheet waits for the promoted runs. |
| 15 | not done | Model cards, validation reports, promotion gates, champion, adapters released | The model cards, the validation reports or the release record is missing. |
| 16 | not done | Site live on Pages with every route; the desk grading live; the live check | results/live_check.json is missing: the site and the desk are not checked live yet. |
| 17 | done | RESULTS, README with the matrix, DECISIONS, notebooks, AGENTS.md, skills, SPEC, ADRs | DECISIONS.md has 23 dated entries, 3 of them reversals; 3 notebooks; 7 ADRs; 3 skills; AGENTS.md and docs/SPEC.md are in place. |
| 18 | not done | Coverage, types, lint, gates, forty commits, rederive, v0.1.0, licenses, repo described | 129 commits on main when this manifest was built; the gates (dashes, vocabulary, palette, skills, tokens, identifiers, types, lint, coverage at 80 percent) run in CI on every push; missing: the rederive; repository topics; the v0.1.0 tag. |

## The questions

The generator drew 10,664 candidate questions from the templates and the catalog, kept a pool of
5,731 after the filters, and split them by entity tuple and template. readback's own golden set gives
the readback test and development sets.

| Set | Questions |
|---|---:|
| train | 2,400 |
| dev | 240 |
| test | 377 |
| template_test | 156 |
| readback_test | 84 |
| readback_dev | 40 |

By family:

| set | lookup | comparison | trend | top_k | multi | refusal |
|---|---:|---:|---:|---:|---:|---:|
| train | 480 | 432 | 360 | 336 | 552 | 240 |
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
audit ran. The leaked defect arm shows sample rows that include the answer row on 927 training
prompts (38.6%), the share where such a row exists at the asked grain.

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

The probes have come back on results branches and are not promoted to main yet, so their numbers
are not in the manifest. DECISIONS.md records what each probe found and what changed because of it.

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
