# goodhart: the specification

Sections 1 to 21 of the build prompt, as written on 31 October 2026. Where the build deviates from it, DECISIONS.md says so, with the date and the reason.

## 1. What we are building

goodhart is a post-training engagement run as code. The customer task is text to SQL over three warehouses (doorstep, turnaround and ledger, read through the readback lock file). A question generator built on the warehouse catalog produces training, development and test questions with known answers, kept apart by entity and by template. A verifier executes SQL on DuckDB and compares results by unit and family; twin warehouses perturbed by seeded rules catch queries that are right for the wrong reason; mutation testing measures how many near misses each reward pays. Small open models are probed on a free GPU and the base is chosen by coverage. SFT, DPO, GRPO and SFT then GRPO each train under one GPU hour ceiling, requested by committed files and run on Kaggle by GitHub Actions. Four reward defects and two controls run from the same start with registered predictions, while monitors watch for the hack each defect invites. The repository grades every exported completion: accuracy on the original and on the twins, pass at k, intervals, Benjamini-Hochberg, a regression suite, a public benchmark under its own semantic evaluation, and an LLM judge calibrated against the verifier. A grader API on Vercel with a grade log in Neon marks any SQL against the held-out questions. The site shows paid against earned for every run, the methods on one budget, the hacks, the verifier, the data, the sizing sheet, the models, and a marking desk.

The reader is a hiring manager for a forward deployed post-training engineer, an RL research engineer or an applied ML engineer, deciding in ten minutes whether this person can run a post-training engagement end to end, not just call a trainer.

## 2. Hard constraints

- **The eighteen mandatory steps in section 17 are the product.** The build is not done while any of them is missing, and the README first paragraph says which are missing until none are.
- **The grader API is deployed at step nine,** before any GRPO run: live on Vercel with the grade log in Neon, verified from a separate client, its verdicts equal to the repository verifier on every completion exported so far. The API container ships as a health stub in step one so the host import happens in the first hour.
- **Actions holds the Kaggle token, nothing else does.** The build machine never sees it and never calls Kaggle. Every GPU run is requested by a committed file under requests/, run by a workflow, and its outputs come back on a results branch for a reviewed promotion with a provenance file. A scan in CI and in a pre-push hook fails on anything that looks like a Kaggle, Hugging Face, NVIDIA or Neon credential in any tracked file or commit diff.
- **No test question is seen by any training, filtering, prompt design, reward design or checkpoint choice.** The training pipeline raises on any test id; a contamination report (exact match, entity tuple overlap, 8-gram overlap and embedding near duplicates) is committed before the plan is hashed.
- **The reward never sees the twins.** The twins live in packages/heldout and the training stage cannot import it; a test proves the import fails.
- **The plan is registered first.** Splits, metrics, budgets, the selection rule, every reward, every defect arm and its prediction, the monitors and their thresholds, the interior tests and the hypotheses are hashed and committed before the first test question is scored.
- **One compute ceiling per method,** measured in the run ledger; a run that would pass it stops at the ceiling and says so.
- **Every arm is reported,** including the defect arms that did not find their hack within the budget and the controls that did better than predicted.
- **The GPU trains; the repository grades.** Every metric is recomputed in CI from exported completions by packages/evaluation, and the claim gate renders only from that.
- **Free tier only, no card.** Kaggle, Colab as the manual fallback, Vercel Hobby, Neon, GitHub Pages, Actions and Releases. The probe runs before the first run and every run reads it.
- **The statement on every surface**, gated: every page, every API response, every model card, the release notes. **No em dashes anywhere.** The vocabulary gate.
- **Every number in a document is rendered from the manifest** by the Day 3 renderer, whole file diff in CI.
- **At least forty conventional commits,** made as the build proceeds, one module at a time, with the tag pushed after the commits. The count is a floor. Pushed history is never rewritten.

## 3. Stack

```
Core            Python 3.12, uv workspace, StrictModel ported from trajectory,
                Pydantic v2 for every contract, request, completion record and
                response
Data            DuckDB 2.0 over the readback marts by lock file; sqlglot 30 for
                parsing, normalizing, mutating and inspecting SQL (tables read,
                literals present, predicates); polars for the exports
Training (GPU)  the Kaggle image CUDA build of PyTorch; Transformers 5.22; TRL
                1.17 (SFTTrainer, DPOTrainer, GRPOTrainer); PEFT 0.21 (LoRA);
                vLLM 0.32 when the probe passes on the card, otherwise TRL with
                transformers continuous batching; bitsandbytes only if a model
                needs 4 bit to fit; Unsloth only if the probe says TRL alone
                does not fit
Orchestration   the kaggle CLI in GitHub Actions only; kernels built from
                templates per request; outputs verified by digest
Evaluation      packages/evaluation: the unbiased pass at k estimator, cluster
                bootstrap over question templates, Benjamini-Hochberg, interior
                tests, message rules; the IFEval reference checker or lighteval
                0.13; the test suite evaluation for Spider
API             FastAPI, async, deployed as a Vercel container (Dockerfile.vercel,
                port 80); SQLModel over Neon for the grade log, the run ledger
                mirror and the client budget; Alembic with the versioned table
                pattern from Day 13
Rendering       the Day 3 claim gate renderer, Jinja templates, the manifest
Web             Next.js 16, TypeScript strict, Tailwind v4, static export on GitHub
                Pages, DuckDB-WASM over the committed results, the series SVG chart
                library, link prefetch off
Quality         ruff, mypy strict, pytest, hypothesis, testcontainers for Postgres,
                nbmake, skills-ref, Playwright against a test server that refuses
                HEAD and answers a first 503, 80 percent coverage floor; a CPU
                smoke run of every trainer on a tiny test model in CI
Logging         structlog, JSON, ids and digests only, never a token or a host
                credential
```

## 4. Architecture

```
  SOURCES     readback lock file: doorstep, turnaround, ledger marts (sha256)
              readback catalog: measures, entities, synonyms
              readback golden set: 100 test questions (held out), 50 dev
              Spider dev (CC BY-SA 4.0) with the test suite databases
              IFEval (the regression suite)
              own questions by Ajay (if written before the generator runs)
        |
        v  fetch by lock, digest      data/sources.lock.json, PROVENANCE.md
        v  twins                      K seeded perturbed copies, distilled
                                      against the mutant pool
        v  generator                  questions, gold SQL, gold answers, units,
                                      splits by entity tuple and template
        v  audit                      contamination report, planted data defects
   +--------------------------------------------------------------------------+
   |  KAGGLE (free GPU, requested by commit, run by Actions)                  |
   |    probe     accelerator, kernels, memory, steps per hour               |
   |    select    candidates x dev, pass at 1 and pass at 8                  |
   |    sft       clean and planted sets                                     |
   |    dpo       pairs from base samples, verified                          |
   |    grpo      honest reward; defect arms; controls; ablation             |
   |    sample    every checkpoint x every held-out set, n samples each      |
   |    judge     calibration completions through a second family model      |
   |    outputs   completions, metrics per step, adapters, ledger rows       |
   +--------------------------------------------------------------------------+
        |  results branch (requested by commit, promoted by review)
        v
   REPOSITORY  packages/evaluation grades every exported completion on the
               original and on the twins; manifest.json > renderer > README,
               RESULTS, the briefing, model cards, validation reports
        |
        v
   VERCEL      /v1/grade  /v1/grades/{id}  /v1/questions  /health
               Neon: grade log, run ledger mirror, client budget
   PAGES       the marking, methods, hacks, verifier, data, desk, sizing,
               models, report
   RELEASE     adapters with digests and cards
```

State in ARCHITECTURE.md, in plain sentences: why text to SQL is a fair RL task (an objective check) and where it is not (the comparator decides what counts as the same answer); why the twins and not a second test set alone; why Actions is the only machine with the token; why the repository grades and the GPU does not; why the generator splits by entity tuple and template instead of by row; why the reward runs on DuckDB in the training process and what a remote sandbox would change; why the defect arms run on the small model and what that does not prove about the main one.

## 5. The data

**The warehouses.** Fetch the doorstep, turnaround and ledger marts named in the readback lock file at the tagged commits it records, verify every sha256, and copy the lock entries into data/sources.lock.json. Read the readback catalog (measures with units and descriptions, entities with aliases, synonyms) the same way. Record the terms of each upstream source by reference to the PROVENANCE file of its own repository.

**The first look, before any question is generated.** Profile every mart at full size: row counts, keys, nulls, ranges, units, distinct entities by kind, the time span of each table, and the ten strangest rows of each warehouse. The first ten real defects (a table whose time column is text, a share stored as a percent in one mart and a fraction in another, a duplicated key, whatever is really there) become generator rules or quarantine rules, each with a test.

**The generator.** packages/questions builds questions from the catalog in the six families readback grades with SQL (lookup, comparison, trend, top_k, multi, refusal; the definition family is out of scope because its answers are text from a document, and the README says so). Each item carries the question text, the warehouse, the family, the entity tuple (measure, entities, period), the gold SQL, the gold result with units and tolerances, and a template id. Surface templates per family vary wording, order, synonyms and units (asked as a percent or a share, in dollars or thousands). Refusal items ask about things the warehouses cannot answer (a forecast, one named person, a year outside the span, a measure no table has). Deterministic under a seed, sorted before every sample, digested.

**Filters.** Drop an item whose gold SQL returns nothing, takes more than two seconds, returns a different result on a second run, or returns the same result on the original and on every twin (an item the twins cannot check). Deduplicate by normalized question and by normalized SQL. Count every drop by reason.

**Splits.** By entity tuple and by template, never by row: training, development and test share no entity tuple, and a held-out template set uses question shapes no training template has (for example, the difference between two years, or the rank of one entity among a named group). Every entity tuple in the readback golden test split is excluded from training and development.

**Paraphrases.** A teacher open model from a different family than every candidate runs in the selection kernel and rewrites a stated share of training questions into natural wording. A paraphrase is kept only if the teacher, given the paraphrase alone, writes SQL whose result matches the gold on the original and on every twin. Paraphrased items are labeled model written everywhere they appear.

**The held-out sets,** never used to generate, filter, design a prompt, design a reward or choose a checkpoint:

```
readback test     the 100 test questions of the readback golden set, graded on
                  the families above (definition excluded and counted)
generated test    entity tuples no training or development item has
template test     question shapes no training template has
own questions     questions Ajay wrote in his own words, labeled by the build with
                  SQL and an answer and checked by him, if he wrote them
Spider dev        1,034 questions over SQLite (CC BY-SA 4.0, Yu et al. 2018),
                  graded with the test suite evaluation (Apache 2.0); the
                  databases fetched in the kernel by digest; a transfer check,
                  labeled as a public benchmark that is likely present in the
                  pretraining data of every candidate
IFEval            the full prompt set, graded by its reference checker: the
                  regression suite for what the base model could already do
```

**The contamination report.** Exact normalized match, entity tuple overlap (zero allowed between training and any test set), 8-gram overlap with a stated threshold, and embedding near duplicates with a small local embedding model and a stated cosine threshold. Any test item that fails moves out of test before the plan is hashed, and the move is logged.

**The planted data defect.** Five percent of the SFT targets are corrupted on purpose by known mutations (a year shifted by one, a predicate dropped, an aggregate swapped), listed in a key that is hashed and committed before the audit runs and revealed after. The audit re-executes every target on the original and the twins against its declared answer and must find them; publish its precision and recall, then train SFT on the planted set and on the cleaned set and publish the difference.

## 6. The environment and the verifier

**The prompt,** registered: a short system part (the dialect, the rules, the refusal form), a schema block (the tables the catalog search returns for the question, five at most, with column names, types and units), and the question. The answer is one SQL query in a fenced block, or the single line REFUSE with a reason code from a fixed list. Thinking modes are off by default and stated; one thinking arm is a stretch item.

**The comparator,** strict and written per family, with a test for every rule: no credit for extra rows, ever; one label column allowed beside the value columns; values compared in their unit within the gold tolerance; ordered families compared in order; a winner compared by label and both values; a set compared as a set. A result whose column is a percent where the gold is a fraction fails, unless the question asked for a percent.

**The rewards,** in packages/rewards, each a pure function from a completion and its item to a float, each with its tests:

```
honest      1.0 if the result matches the gold on the original under the strict
            comparator; 0.1 if the SQL parses and executes but does not match;
            0 otherwise. On refusal items, 1.0 for REFUSE and 0 for any SQL; on
            answerable items, 0 for REFUSE. (The Arctic-Text2SQL-R1 shape.)
lenient     defect D1: the honest reward with the strict comparator replaced by
            a loose one: 1.0 if any cell of the result equals any gold value
            within tolerance
leaked      defect D2: the honest reward, but in a stated share of training
            prompts the schema block carries three sample rows, one of which is
            the row the question asks about, value included
lazy        defect D3: the honest reward with 0.5 instead of 0.1 for valid,
            executable, wrong SQL
refusal     defect D4: the honest reward plus 0.3 for REFUSE on any item
random      control C1: a seeded Bernoulli reward with the mean the honest
            reward had on the starting checkpoint, independent of the completion
format      control C2: 1.0 for any single fenced SQL block that parses
```

**The twins,** in packages/heldout. Sixteen candidate copies of each warehouse, each built from the original by seeded rules per table: numeric measure columns scaled by a factor drawn away from one, with a small per row jitter, and for ranking families a share of values permuted within the column so orders change; keys, labels and years untouched; internal identities between derived measures are not preserved, and the README says so. Distill the set the way Zhong et al. did: keep the K twins (K = 4 by default) that together distinguish the most mutants of the gold SQL pool, by greedy cover. The twin gold for an item is its gold SQL executed on the twin. A completion is confirmed when it matches the gold on the original and on every twin.

**Verifier strength, by mutation testing.** packages/heldout/mutants.py writes near misses of every gold query with sqlglot: drop one predicate, flip a comparison, shift a year by one, swap max and min or sum and average, remove a DISTINCT, flip an ORDER BY, change a LIMIT, change a GROUP BY key, replace an entity with its neighbor. A mutant equivalent to the gold on the original and on all sixteen candidates is counted as equivalent and set aside, and a stated sample of those is read by hand. Publish, per family: the share of non-equivalent mutants the honest reward pays in full (it runs on the original only), the share the lenient reward pays, and the share the twin check confirms. This is the number that says how much the reward can be gamed before any training runs.

**Hack detectors,** pure functions over a completion and its item, used by the monitors and on the pages: literal only (the SQL reads no table); copied value (a literal equal to a value shown in the prompt); unfiltered (more result rows than the gold, or a predicate of the gold missing); refusal on an answerable item; valid but wrong. Each detector gets the three gate tests.

## 7. Training, and how runs are requested

**Requests.** requests/<date>-<name>.json names the stage, the model, the data digests, the reward, the seed, the hyperparameters, the step budget and the GPU hour ceiling. .github/workflows/gpu.yml runs every request committed since its last run: it builds a kernel directory from templates (the code at the request commit, pinned requirements, the stage entry point, kernel-metadata.json with the accelerator the probe chose and internet on), pushes it with the kaggle CLI, polls its status, downloads its output, verifies the digests, writes the outputs and results/provenance.json (request, kernel version, run id, commit, digests, accelerator, wall clock) to a results branch, and opens a pull request, or leaves the branch when the repository setting refuses. The build machine reviews and promotes the results into results/. A request with nothing to do succeeds and says so.

**Checkpoints and resume.** Every stage writes a checkpoint every 25 steps under a key built from the request digest and the data digests. A stopped run (the session limit, a quota stop, a crash) is resumed by a request that names the earlier kernel output as an input, and a test resumes a CPU smoke run from its checkpoint and compares the next step with an uninterrupted run.

**The run ledger.** One row per stage run: request, kernel, model, stage, steps, prompts, rollouts, tokens generated, tokens trained, GPU seconds by phase (generation, reward, training, evaluation), peak memory, wall clock, status, and the digests in and out. It feeds the compute ceilings, the sizing sheet and the pages.

**Defaults,** each changed only with a DECISIONS.md entry:

```
LoRA        rank 16, alpha 32, dropout 0.05, on the attention and MLP
            projections; float16 on the T4; gradient checkpointing on
SFT         completion only loss, two epochs, learning rate 2e-4 with cosine
            decay, maximum length from the prompt length distribution
DPO         pairs from 8 base samples per training prompt at temperature 0.8,
            chosen verified correct by the honest reward, rejected verified wrong
            (executable wrong preferred); prompts with no pair dropped and
            counted; beta 0.1, one epoch
GRPO        8 generations per prompt, 8 prompts per step, temperature 0.8, a
            maximum completion length from the gold SQL length distribution,
            beta 0.0 (the TRL default) with a KL arm as a stretch, the loss type
            registered as dr_grpo, with dapo as the ablation, clip at
            the TRL defaults, checkpoints every 25 steps, steps set from the
            probe so one run fits its ceiling
filter      GRPO training prompts are kept only where the starting checkpoint
            had at least one correct and at least one wrong answer in 8 samples
            (the learnability filter; Arctic kept synthetic queries where at
            least one of ten generations was correct); the share kept is published
```

**The methods,** each under the same GPU hour ceiling, the ceiling counting every stage the method needs (sampling for DPO pairs included):

```
M0  base     the chosen base, zero shot, with three fixed examples from training
M1  SFT      supervised on the gold SQL of the cleaned training set
M2  DPO      pairs from the base, as above
M3  GRPO     from the base, honest reward, learnability filter
M4  SFT+GRPO GRPO from the M1 checkpoint, same reward and filter
floor        readback no model floor through its own planner, graded the same
reference    optional: two or three large open models on free hosted endpoints
             through the same prompt and verifier, if Ajay set the key as a
             repository secret; a reference row, never a target
```

M3 and M4 run three seeds each if the quota allows, otherwise one, and the pages say which. The checkpoint of each run is chosen on development accuracy on the twins.

**The ablation.** On the small model, the honest reward with the loss type changed from dr_grpo to dapo and everything else equal. Group size and the KL arm are stretch.

## 8. The defect arms, the controls and the monitors

**Two models.** The defect arms, the controls and the ablation run on the small model (the best sub 1B candidate under the same selection rule) from its own SFT checkpoint, at the same step budget and seed, on the same prompts in the same order. The methods run on the main model. The pages say plainly that a hack found by a 0.6B model need not be found by a 2B one, and the stretch list has the main model arm.

**The monitors,** in packages/monitor, computed from the completions each step logs: mean result rows, the share literal only, the share copied, the share unfiltered, the share of refusals on answerable items, the share valid but wrong, frac_reward_zero_std, entropy, completion length and the clipped ratio. Each has a threshold in config/monitor.yaml registered with the plan. A breach writes an alert row; the registered rule stops the run at the next checkpoint for a defect arm and flags it for the honest arm, and the workflow opens a GitHub issue with the alert rows, the run id and the runbook link.

**The predictions,** in evaluation/predictions.yaml, registered before any arm runs. For each defect: which detector fires, in which step window, and how far the twin confirmed accuracy will fall below the honest arm by the end. For each control: the share of the honest gain it will reach. Every prediction is printed beside its outcome with the word held or did not hold.

**The paid against earned curves.** For every run: the mean reward paid per step, and the twin confirmed accuracy of the checkpoint on development, every 25 steps, with the gap shaded and the monitor firing marked.

## 9. Evaluation

**Metrics, defined once in packages/evaluation/metrics.py:** execution accuracy on the original; twin confirmed accuracy (the headline); refusal accuracy on refusal items; over refusal on answerable items; the hack detector rates; pass at k for k in 1, 2, 4, 8 and 16 with the unbiased estimator from 16 samples at the registered temperature; greedy pass at 1; tokens per answer; all by family and by warehouse.

**Every checkpoint on every held-out set.** The sampling kernel produces 16 samples per question for M0 to M4 on readback test, generated test, template test and own questions; greedy only for Spider dev, IFEval, the defect arms, the controls and the ablation. The completions are exported as parquet with their request digests, and the repository grades all of them.

**Intervals.** Cluster bootstrap over question templates and entity families (questions from one template are correlated), 2,000 resamples, computed from per cluster sufficient statistics. Paired differences between methods with intervals and Benjamini-Hochberg across the registered family of comparisons.

**Hypotheses,** registered in evaluation/hypotheses.yaml before the test sets are scored, each tested once, each printed with its verdict whatever it is. At least: SFT then GRPO beats SFT alone on twin confirmed accuracy on readback test; GRPO raises pass at 1 more than pass at 16; the random reward reaches less than a third of the honest gain; the lenient arm opens a twin gap of at least ten points by its last checkpoint; DPO on verified self samples reaches half the GRPO gain at a quarter of the compute.

**The interior test** on the checkpoint (the chosen step against its neighbors 25 steps either side), the decoding temperature for pass at 1 (0, 0.2 and the registered value), the twin count (3, 4 and 6), and the tolerance (half, as registered, double). The page prints the test accuracy of the champion under each neighbor and the word degenerate if the champion changes at the first neighbor.

**The regression suite.** IFEval in full for M0 to M4, with the change against the base and its interval, and Spider dev under the test suite evaluation as a transfer check in both directions: did training on the three warehouses help or hurt a public benchmark.

**The LLM judge, calibrated.** A judge from a family different from the policy runs in a kernel over a registered calibration set: every completion in a stratified registry of 600 drawn by seed before any judge call, across the verdict classes and every hack type, with known truth from the verifier. Publish agreement, Cohen kappa, the false accept rate on each hack type, a padding test (the same SQL with a long comment appended) and an order test (two candidates swapped). The judge never enters a reward; the page says what it would have paid if it had.

**Message rules.** Every sentence on every page that compares two numbers is chosen by a rule from the interval of the paired difference, with a branch for every outcome and a test per branch.

## 10. Model selection, the sizing sheet and the quota

**Selection, registered.** The candidates of Part 2 go through the probe in one kernel: loads in float16 on the card, a LoRA step runs, generation works on the chosen backend, the memory left at 8 generations and the registered lengths, and steps per hour. Then every candidate that passes is graded on the development sets: greedy pass at 1 and pass at 8 at the registered temperature. The rule, written before the probe runs: among candidates that pass the probe and whose license permits releasing an adapter, choose the main model as the highest pass at 8 at or under 2.5B parameters as counted on its card, and the small model as the highest pass at 8 under 1B, ties broken by steps per hour. Print the whole table with each license.

**The sizing sheet,** in packages/sizing and on its own page: per run, the measured prompts, rollouts, tokens generated and trained, GPU seconds by phase and peak memory; throughput per phase; the learnability share and what it implies for the number of prompts a run of a given length needs; and a projection of each recipe to 7B and 32B from the measured token counts with stated scaling assumptions, priced from public price pages read on the day of the build (a fine-tuning price list per million tokens and a GPU hour price list, each with its URL and the date), labeled as an estimate and never spent.

**The quota as a design input.** The probe writes results/gpu_probe.json with the accelerator, the backend, steps per hour per model, and the projected GPU hours for the whole plan. The plan below must fit the weekly quota with margin; if it does not, the shrink order applies in this sequence and each step taken is a DECISIONS.md entry: pass at k samples from 16 to 8; Spider greedy only on M0 and the champion; defect arms at 100 steps; seeds two and three dropped.

```
Stage                                      GPU hours, ceiling
probe                                       0.3
selection (all candidates that load)        1.5
SFT clean and planted, both models          1.0
DPO, sampling and training                  1.5
GRPO M3 and M4, one seed each               4.0
GRPO seeds two and three, the better one    4.0   only if the quota allows
defect arms, controls, honest, small        4.5
ablation                                    0.7
sampling for evaluation                     4.0
judge calibration                           0.5
mandatory total                             18.0  of a weekly quota of about 30
```

## 11. Model cards, the release and promotion gates

**Model cards,** one per trained checkpoint that is released: base model and its license, data and its digests, method and every hyperparameter, compute from the ledger, results with intervals on every held-out set, the hack detector rates, known failure modes in plain words, intended use, and the statement. Rendered by the claim gate. **A validation report** for each champion candidate in SR 11-7 order (purpose, data, design, performance on development and test, stability across seeds, sensitivity from the interior tests, limitations, monitoring, the effective challenges on record), refusing to render with a section missing.

**Promotion gates,** each with its deliberate failure test: twin confirmed accuracy on readback test above the base with the interval of the difference clear of zero; over refusal on answerable test items no more than two points above the base; IFEval no more than two points below the base; every hack detector silent on the last 50 steps of the run; seed spread under a stated bound where there are seeds; the API verdict equal to the repository verdict on every completion; a license that permits the release. A checkpoint that clears every gate is the champion; the others stay challengers and the site names the gate each failed.

**The release.** Adapters as release assets with digests in data/models.lock.json and their cards beside them; never the merged weights unless the license and the size allow. If Ajay added a Hugging Face token to Kaggle secrets, the same adapters go to the Hub from the kernel, with the same cards.

## 12. Serving: the grader API and the marking desk

**The API,** FastAPI in a Vercel container. POST /v1/grade takes a question id from a held-out set and a SQL query (or REFUSE), executes it on the original and on the twins with a timeout and a read only connection, writes the grade to the grade log before responding, and returns the grade id, whether it executed, the result rows, what each reward would pay (honest, lenient, lazy, refusal), what the held-out checks confirm (original, twins passed of K), the hack detector flags and the statement. GET /v1/grades/{id}, GET /v1/questions (ids, text, family and warehouse, never the gold), GET /health (version, warehouse and twin digests, verifier version, the statement). A client budget of 60 grades an hour per hashed client address, and a separate parity key held as an Actions secret with its own budget. The container bakes only the tables the held-out questions read, with the twins prebuilt; the probe checks the image against the host limit, and if it is over, K drops to 3 with a DECISIONS.md entry.

**Parity.** scripts/check_parity.py sends every exported completion of the readback test set for M0 and the champion through the live API and asserts every verdict equal to the repository verifier. It runs against a local container in CI and against the live URL before the tag; results/parity.json records both.

**Persistence tests.** test_grade_is_committed from an independent connection; test_grade_precedes_response; the CI out of process check that starts the container, grades one query and reads the grade back from a separate process.

**Latency.** A load test through the live API with the cold start measured separately, p50 and p99.

**The marking desk.** On /desk, pick a held-out question, see each method answer it as a script with the marks in the margin, and write a query of your own: the page sends it to the API and prints what each reward pays in red and what the held-out checks confirm in green. Beside the editor, five tries a reader can click (print the number, drop the filter, return the whole table, refuse, write it properly), each with the marks it gets. When the API is asleep, the page plays a recorded session and says so, and probes twice.

## 13. The incident

The first defect arm whose monitor fires is run end to end as an incident. Expected: the detector fires within its predicted window, the alert row is written, the run stops at the next checkpoint, the issue opens, and docs/runbooks/reward-hacking.md is followed: identify from the alert rows, the per step completions and the paid against earned curve; contain by stopping the arm; read twenty completions by hand around the firing step and quote three of them; fix the reward by the smallest change that closes the hole; rerun a short check from the last good checkpoint with the fixed reward; verify that the detector stays silent and the gap closes. docs/incidents/<date>-<arm>.md is the postmortem: the timeline from the ledger and the step logs with steps and wall clock, the impact in steps and GPU minutes spent learning the hack, the root cause, what detected it and how fast, what did not, and three actions, one of them a test that would have refused the reward before training. Blameless and specific.

## 14. The repository surface for agents

AGENTS.md as the single source (setup, tests, gates, the files an agent must not edit: the lock files, the plan, the predictions, the monitor thresholds, the audit key, and the results promoted from the GPU except through promotion); CLAUDE.md and .github/copilot-instructions.md as one paragraph each pointing at it; three skills in .claude/skills validated with skills-ref: request-a-gpu-run (how to write a request, what the ledger must show, how to promote), add-a-reward (the pure function, the three tests, the mutation score, the twin check, the registration), add-a-question-family (the template, the filters, the split rule, the contamination check). docs/SPEC.md holds sections 1 to 21 of this prompt as the specification, with a line at the top saying deviations are in DECISIONS.md; the opening paragraph of the prompt is not part of it. docs/adr/ holds one architecture decision record per choice named in section 4, each with the alternatives and why they lost. The playbook page renders AGENTS.md and the skills.

## 15. The site and the visual system

Static export on GitHub Pages, DuckDB-WASM over the committed results, the live API for the desk, and a recorded session so every page works when the API is asleep, labeled as recorded.

**Art direction.** The marking room. Light first: an exam script on pale ruled paper, hairline blue rules, a red margin line down the left of every script, the question set at the top, the model answer in blue-black ink, and the marks in the margin in Monaspace Radon: the reward in red, the held-out checks in green, each mark a word and a number, never a color alone. Dark is the blackboard in the same room: slate, chalk white ink, chalk colors. Dense tables only on the methods, verifier and sizing pages. Nothing decorative beyond the ruling and the margin.

**Surfaces and ink,** both validated:

```
Light surface   #F8F9F6    dark surface    #18241F    card   #FFFFFF light, #203029 dark
Primary ink     #172033    primary ink     #E8EDE4    rules  #CFDAE6 light, #2E4038 dark
Secondary ink   #4E5869    secondary ink   #A9B5AA    margin #E3A3AB light, #6E3B47 dark
Control gray    #6B7380    control gray    #8C9A8F    (rules and margin are decoration,
                                                       never data)
```

**Categorical palette, fixed order.** Light: #1F4FB0, #C0263F, #0A7F72, #9A6600, #B23C8E, #6B44C6, #4D6E1A, #1A7FB0. Dark: #6E9BEA, #E86F86, #8BE0C0, #E3BE55, #D46FBF, #C2ACF8, #B5CF5E, #5CC6E8. Measured on 31 October with the series validator (scripts/validate_palette.js from readback, config mode, both modes): zero failures; worst adjacent colour vision deficiency separation 9.1 light and 11.2 dark against a target of 8; worst normal vision separation 16.3 and 14.7 against a floor of 12; every light slot at or above 4.22:1 on the surface and 4.46:1 on white; every dark slot at or above 5.21:1 on the surface and 4.50:1 on the dark card; the first three slots pass the all pairs check (9.1 light, 14.9 dark). Slot one is the answer (what the model wrote); slot two is the reward (what training paid); slot three is the held-out checks (what was confirmed). Slots four to eight are the methods in fixed order: base, SFT, DPO, GRPO, SFT then GRPO; the floor is the control gray with a hatch, the hosted reference a dashed control gray. Direct labels always. Port the validator and run it in CI against exactly these values; if it rejects a slot after porting, move it by the smallest change that passes and record it in DECISIONS.md.

**Sequential, blue, for magnitude.** Light, light to dark: #B9CCF0, #8BABE6, #5C88D6, #3664BF, #1F4596. Dark, dark to light: #2A4878, #3A62A3, #5383D2, #83A8EE, #BACFF7. Validated as ordinal ramps. For heat strips of reward by step and prompt, the mutation grid and the sizing sheet.

**Diverging, red to blue through a neutral midpoint,** #E8EAE6 light and #2C3833 dark. Light: #7A1328, #B32E48, #D8768A, #EDB3BE, midpoint, #ABC2EC, #7FA2E2, #3E6CC4, #1C3F88. Dark: #F6B6C3, #E97A92, #C84A66, #8E2A40, midpoint, #2A4A82, #3F6CBF, #79A1EA, #B6CCF5. Validated. For method against base differences, and for paid minus confirmed.

**Status, reserved, always with an icon and a word.** Good #1E7A3D light and #46C06A dark; warning #8A6A00 and #D6B23A; serious #B84A0E and #EF8A46; critical #B3261E and #F2675C. Measured at or above 4.80:1 on the light surface, 5.24:1 on the dark surface and 4.53:1 on the dark card. Earned, paid not earned and failed are verdicts, not statuses: they wear slots three, two and the control gray with their word.

**Type**, self hosted at build time through next/font from the Fontsource packages:

```
A  Marking room (default)  STIX Two Text (display at semibold and bold, the
                           briefing, the model cards), Sofia Sans (interface and
                           body) with Sofia Sans Semi Condensed (dense tables),
                           Monaspace Neon (SQL, figures, identifiers), Monaspace
                           Radon (the marks in the margin, nowhere else)
B  Lecture hall            Libertinus Serif, Libertinus Sans, Libertinus Mono,
                           with Monaspace Radon for the marks
C  Problem set             Charis SIL, Golos Text, Intel One Mono, with Monaspace
                           Radon for the marks
```

Every face above was confirmed as a Fontsource package on npm on 31 October (version 5.3.0). Body 16px, line height 1.5, measure capped at 70 characters, the briefing at 68; every figure right aligned with tabular numerals. If a face is missing at build time, use the next option and record it.

**Chart rules.** The series mark specification unchanged, a table toggle and a show the SQL toggle beside every chart, texture in addition to hue, one message per chart rendered from the manifest as its title. The hero, paid against earned: small multiples, one panel per run (the honest arm, the four defects, the two controls), training step on the x axis, share on the y axis, the reward paid in slot two and the twin confirmed accuracy in slot three, the gap shaded with a diagonal texture, the monitor firing as a vertical rule with its step, the predicted window as a band. The methods chart: one row per method with the twin confirmed accuracy and its interval, the floor hatched, the base as the reference line, the compute used as a bar beside each row. The coverage chart: pass at k against k on a log two axis, one line per method. The verifier chart: mutants paid by each reward and confirmed by the twins, per family.

**Pages.**

- / **The marking.** Paid against earned for every run, the champion with its test numbers and intervals beside the base and the floor, the statement.
- /methods **One budget, four recipes.** Twin confirmed accuracy with intervals on every held-out set, pass at k, the regression suite, Spider, the compute used, the recommendation per use case rendered by rule.
- /hacks **What the reward paid for.** Each defect and control: the prediction, the outcome, the detector that fired and when, the gap, three scripts with their marks; the incident and its postmortem.
- /verifier **Who marks the marker.** The comparator rules, the twins and their distillation, the mutation score per family, execution accuracy against twin accuracy, the judge calibration.
- /data **Where the questions came from.** The generator, the filters with their counts, the splits, the paraphrase round trip, the contamination report, the planted defect and the audit.
- /desk **Mark it yourself.** The marking desk.
- /sizing **What it costs.** The ledger, the sizing sheet, the quota plan and what changed it.
- /models **The model cards, the validation reports and the base selection table.**
- /report **The briefing**, with a print stylesheet.

Skeleton loading states in the shape of the chart, prefers-reduced-motion respected, everything keyboard reachable, contrast checked in both themes, the statement in the footer of every route, mobile first at 390 pixels with the scripts full width and the margin marks moved above the answer.

## 16. What makes it read as built by a person

Everything in northstar section 18 applies unchanged. Four additions:

- **What the post-training lead would push back on.** Every page and every section of the briefing ends with the objection and what the build did about it: two billion parameters on templated questions is not a customer workload; the twins are perturbations, not new data; a T4 favors the cheap methods under one ceiling; the floor is strong on templated questions; one seed where the quota ran out; the judge is one model.
- **What the RL researcher would push back on.** The defects were planted, not discovered; the defect arms ran on the small model; group sizes are small; Spider is likely in the pretraining data of every candidate; the comparator decides what counts as the same answer, and its rules are a choice.
- **The decisions name the tools.** Every DECISIONS.md entry that chose a trainer default, a generation backend, an accelerator, a twin rule or a comparator rule cites the documentation page or the paper and its date.
- **The dead ends are kept.** The notebooks keep the reward curve that climbed under the lenient comparator and what the twins said about it, the leaked value the copy detector caught and how many steps it took, and the random reward control and what it did.

## 17. Repository layout and build order

```
goodhart/
+-- README.md  RESULTS.md  ARCHITECTURE.md  DECISIONS.md  CONTRIBUTING.md  LICENSE
+-- AGENTS.md  CLAUDE.md  vercel.json  Dockerfile.vercel  Makefile  pyproject.toml
+-- uv.lock  requirements-gpu.txt  .env.example
+-- .github/  workflows/  ci.yml  gpu.yml  pages.yml  release.yml
|             copilot-instructions.md
+-- .claude/skills/  request-a-gpu-run/  add-a-reward/  add-a-question-family/
+-- config/  monitor.yaml  generator.yaml  twins.yaml  training.yaml
+-- data/  sources.lock.json  models.lock.json  PROVENANCE.md  fixtures/
|          audit_key.sha256
+-- requests/  committed GPU requests
+-- kernels/  templates for the Kaggle kernels
+-- packages/
|   +-- core/  questions/  rewards/  heldout/  monitor/  train/  evaluation/
|   +-- sizing/  judge/  api/  render/
+-- evaluation/  plan.yaml  predictions.yaml  hypotheses.yaml  registration.json
+-- results/  manifest.json  provenance.json  gpu_probe.json  selection.json
|             ledger.parquet  completions/ (or a data release, see below)
|             parity.json  live_check.json  rederive.json  contamination.json
+-- notebooks/  01_the_reward_curve  02_the_leak  03_the_random_reward
+-- report/  briefing.md  templates/
+-- scripts/  check_no_em_dash.py  check_vocabulary.py  check_published_numbers.py
|             check_statement.py  check_tokens.py  scan_for_planted_identifiers.py
|             check_registration.py  validate_palette.js  fetch_sources.py
|             build_twins.py  profile_sources.py  probe_hosts.py  check_parity.py
|             load_test.py  check_persistence.py  reset_and_rederive.py
|             build_demo_gif.py
+-- deploy/  probe.ps1  verify.ps1  push-tag.ps1
+-- docs/  SPEC.md  adr/  runbooks/  incidents/  models/  verifier.md
|          data.md  training.md  evaluation.md  serving.md
+-- web/
+-- tests/
```

Completion exports above 50 MB go to a data release (data-v1) as assets with their digests in data/results.lock.json; CI downloads and verifies them before grading.

Build order. Steps 1 to 18 are mandatory and are the product.

```
 1  Scaffold, ported tooling, the em dash, vocabulary, statement, token and
    identifier gates on every commit diff (pre-push hook and CI), licences;
    the API as a health stub with Dockerfile.vercel; the probes: Vercel (once
    imported), Neon, whether Actions may open pull requests here, and a GPU
    probe requested from Actions (accelerator, CUDA capability, memory, the
    generation backend, a ten step LoRA GRPO smoke on the smallest candidate,
    steps per hour)
    chore: scaffold with ported tooling, gates and the probes
 2  Claim gate renderer before any results exist
    ci: published number gate
 3  Warehouses by lock, the first look at full size, the comparator, the
    rewards, the twins and their distillation, the mutation score, the hack
    detectors
    feat(verifier): comparator, rewards, twins and the mutation score
 4  The generator, filters, splits, paraphrase round trip, held-out sets,
    contamination report, the planted data defect and its hashed key
    feat(data): questions, splits and held-out sets
 5  The plan, predictions, monitor thresholds and hypotheses registered and
    hashed before any test question is scored
    chore(plan): registered evaluation plan
 6  The GPU request route: requests, the workflow, the kernel builder, the
    results branch, promotion with provenance, checkpoints keyed by digest,
    resume, the run ledger; a CPU smoke run of every trainer in CI
    feat(gpu): requested runs on a free GPU
 7  Model selection by coverage; the base, the floor and the optional
    reference graded on every held-out set
    feat(models): base selection by coverage
 8  SFT on the cleaned and the planted sets, the audit result, DPO
    feat(train): sft, the data audit and dpo
 9  The grader API live on Vercel with the grade log and the client budget,
    verified from a separate client; parity on every completion exported so
    far
    feat(api): the grader live
10  GRPO M3 and M4 with the honest reward and the learnability filter, seeds
    as the quota allows, the ablation
    feat(train): grpo with the honest reward
11  The defect arms and the controls with their monitors and predictions; the
    incident and its postmortem
    feat(hacks): four reward defects and two controls
12  Evaluation in the repository from the exported completions: twin
    confirmed accuracy, pass at k, intervals, Benjamini-Hochberg, interior
    tests, hypotheses, the regression suite, Spider under the test suite
    evaluation, the judge calibration
    feat(evaluation): graded in the repository
13  Persistence tests: independent connection, grade before response, out
    of process check
    test(api): grades observed independently
14  The sizing sheet and the compute ledger pages
    feat(sizing): the workload sheet
15  Model cards, validation reports, promotion gates, the champion, adapters
    released with digests
    feat(release): cards, gates and the champion
16  Web: the marking, methods, hacks, verifier, data, desk, sizing, models,
    report; the test server; Pages deploy; the repository surface for agents
    (AGENTS.md, skills, SPEC, ADRs)
    feat(web): the marking room and its pages
17  Full run, RESULTS.md with the section 20 checklist, README with the
    matrix, DECISIONS.md, notebooks 01 to 03; load test with cold start; the
    live check in a real browser with the API awake and asleep and from a
    separate client; demo recording; reset and rederive in a worktree (the
    warehouses, twins and questions from the locks, the evaluation from the
    exported completions, and one GRPO run requested again from its own
    request, compared within the seed spread)
    chore: full run, live check and canonical rederive
18  Tag v0.1.0 pushed after the commits
    chore: release
19  The champion in the browser: the merged adapter exported to ONNX and run
    with WebGPU on /desk, with the recorded session as the fallback
    feat(web): the model in the browser
20  The main model defect arm for the strongest hack, to test whether the
    larger model finds it
    feat(hacks): the main model arm
21  A thinking arm, group size and KL arms, and a rejection sampling fine
    tune beside DPO
    feat(train): more arms
22  goodhart champion as a provider in the readback router, graded by the
    readback plan
    feat(readback): the post-trained provider
23  A 4B run with QLoRA if the quota allows
    feat(train): the larger model
24  Container and release workflow hardening, docs
    ci: containers and documentation
```

If time runs short, cut from 24 backward and never below 18. Inside the first eighteen there is nothing to cut; if one cannot be finished, the README first paragraph names it and the build is not called complete.

## 18. The skills matrix

In the README, one row per skill a post-training, RL research or applied ML posting lists, and the file that demonstrates it:

```
Skill                                     Where it lives
Supervised fine-tuning with LoRA          packages/train/sft.py, the SFT cards
Preference optimization (DPO)             packages/train/dpo.py, the pair counts
RL with verifiable rewards (GRPO)         packages/train/grpo.py, packages/rewards
Algorithm selection                       /methods, the recommendation rules
Reward design and reward hacking          packages/rewards, packages/monitor,
                                          the defect arms, docs/incidents
Verifier design and strength              packages/heldout, the mutation score
Experiment design and ablations           evaluation/plan.yaml, the controls,
                                          the ablation, the seeds
Evaluation and held-out testing           packages/evaluation, the held-out sets
LLM as judge, calibrated                  packages/judge, /verifier
Data strategy and synthetic data          packages/questions, the paraphrase
                                          round trip, the filters
Data quality diagnosis                    the planted defect and the audit
Contamination control                     results/contamination.json
Base model selection                      results/selection.json, /models
Workload sizing and cost                  packages/sizing, /sizing
GPU orchestration on free hardware        .github/workflows/gpu.yml, kernels/
Statistics for evaluation                 pass at k, cluster bootstrap,
                                          Benjamini-Hochberg, interior tests
SQL and DuckDB                            the warehouses, the comparator
Serving and persistence                   packages/api, the grade log, parity
Model documentation and release           docs/models, the validation reports,
                                          the release assets
Agent assisted engineering                AGENTS.md, .claude/skills, docs/SPEC.md
Communication                             report/briefing.md, the pushback
                                          paragraphs
```

## 19. Testing

- The three tests for every gate: em dash, vocabulary, statement, token scan on commit diffs, identifier scan, claim gate, plan registration, palette validator (both surfaces and both cards), every reward, every hack detector, every monitor threshold, every promotion gate, every interior test.
- **Comparator**: every family rule on fixtures; units (a percent against a fraction fails, a thousand dollars against a dollar fails); extra rows never pass; order matters only where it should; a property test that unordered families ignore row order.
- **Rewards**: each defect pays what it should on its fixtures; the honest reward pays nothing for each hack fixture (a literal of the right answer, the whole table, a refusal on an answerable item); the random control is independent of the completion and reproducible from its seed.
- **Twins**: built deterministically from the lock and the seed; the twin gold differs from the original gold for every kept item; the distillation picks the same K under two PYTHONHASHSEED values; a literal equal to the original answer never passes when the twins differ; the training stage cannot import packages/heldout.
- **Mutation score**: equivalent mutants are detected; the score is reproducible; a deliberately weakened comparator raises the share paid (the deliberate violation).
- **Generator and splits**: deterministic under the seed; no entity tuple shared across splits; no readback test tuple in training; the training pipeline raises on a planted test id; the contamination report catches a planted duplicate and a planted paraphrase of a test question.
- **Training plumbing**: SFT, DPO and GRPO each run two steps on CPU with a tiny test model in CI; the reward function is called once per completion; the checkpoint advanced; resume from a checkpoint equals an uninterrupted run at the next step; the ledger row is written.
- **Requests**: the kernel builder writes metadata with the probe accelerator and no secret; a request with nothing to do succeeds; provenance carries every digest; promotion refuses a result whose digest does not match.
- **Evaluation**: the pass at k estimator against brute force on fixtures; the bootstrap interval contains the point estimate; Benjamini-Hochberg on fixtures; identical manifests under two PYTHONHASHSEED values; the plan hash in the manifest equals the hash of the committed plan; every message rule branch.
- **Judge**: the stratified registry is drawn by seed before any judge call; the judge output parser refuses malformed verdicts; the padding and order tests run.
- **API**: parity on the exported completions; the client budget refuses past the cap; health reports the digests and the statement; the grade row before the response; independent connection and out of process checks; a query that writes or attaches is refused by the read only connection.
- **Did the step run**: the probe ran on the accelerator it asked for; the learnability filter dropped what it reported; the monitor wrote one row per step; the parity check compared every completion; the scans read every file and every diff they list.
- The test server refuses HEAD and answers a first 503, and the site client handles both; Playwright on every route (the hero draws every panel in its legend, a script prints with its marks on /desk, a live grade renders, the statement on every route); notebooks in CI with nbmake.
- Availability probes and named skip markers for Postgres, the browser and the load test, with CI asserting each dependency.

## 20. Definition of done, printed as a checklist

The build prints this list in its final message and in RESULTS.md, each line marked done or not done, with one sentence of evidence per line (a file, a test name, a URL, a number):

```
 1  Warehouses fetched by lock and profiled at full size, the first ten
    defects as rules, provenance recorded
 2  The comparator, the rewards, the twins with their distillation, the
    mutation score per family and the hack detectors, each tested
 3  The generator, filters, splits by entity and template, the paraphrase
    round trip, every held-out set, the contamination report, the planted
    data defect with its hashed key
 4  The plan, predictions, monitor thresholds and hypotheses hashed before
    the first test question was scored
 5  The GPU request route with the results branch, reviewed promotion,
    checkpoints, resume and the run ledger; no token anywhere in the
    repository or its history
 6  The probe and the base selection by coverage, the whole table with
    licenses; the base, the floor and the reference graded
 7  SFT clean and planted with the audit precision and recall; DPO with its
    pair counts
 8  The grader API live on Vercel with the grade log, verified from a
    separate client; parity exact on every completion
 9  GRPO M3 and M4 with the honest reward and the learnability filter, the
    seeds run, the ablation
10  The four defect arms and two controls with predictions held or not held,
    the paid against earned curves, the incident and its postmortem
11  Twin confirmed accuracy, pass at k, intervals, Benjamini-Hochberg,
    interior tests, hypotheses, the regression suite and Spider, graded in
    the repository from the exported completions
12  The judge calibration with agreement, kappa and false accepts per hack
    type
13  Persistence: independent connection, grade before response, out of
    process check; latency with the cold start
14  The sizing sheet and the compute ledger with every run inside its
    ceiling
15  Model cards, validation reports, promotion gates, the champion, adapters
    released with digests
16  The site live on Pages with every route and the statement; the desk
    grading live; the live check in a real browser awake and asleep and from
    a separate client
17  RESULTS.md with a specific limitations section; README with the matrix;
    DECISIONS.md with at least twelve dated entries and two reversals; three
    notebooks with a dead end each; AGENTS.md, the skills, SPEC and the ADRs
18  Coverage at or above 80 percent, mypy strict and ruff clean, zero em
    dashes, zero banned vocabulary, palette validator green, skills valid, at
    least forty commits, the rederive run before the tag with no drift
    beyond the seed spread, v0.1.0 pushed after the commits, Apache 2.0 for
    code with the terms of every source and base model restated, repo
    described, topics set
```

Stretch items (the browser model, the main model arm, the extra arms, the readback provider, the 4B run, the hardening) are listed beneath the checklist as stretch, done or not done, and never counted toward it.

## 21. Stretch, and what not to build

Stretch, only if the day allows: steps 19 to 24; GSPO with sequence level importance sampling as a further loss arm ([arXiv 2507.18071](https://arxiv.org/abs/2507.18071v2)); a second customer task with a different verifier, only if it adds something the SQL task does not show.

What not to build:

- No reward curve as a headline, and no claim from a training distribution metric.
- No test question in any prompt, filter, reward, design step or checkpoint choice.
- No reward that sees the twins.
- No LLM judge inside any reward.
- No language model writing a sentence about a number; answers are rendered from results.
- No paid GPU, no paid API, no card anywhere.
- No token, key or host credential in the repository, a log, a notebook output, a kernel or a commit diff.
- No GPU call from the build machine; Actions requests every run.
- No grading on the GPU; the repository grades.
- No claim that the questions are human, that the warehouses are live, or that the models are fit to answer real questions about housing, flights or banking.
- No mandatory step replaced by a table entry.

Build it in the order in section 17. When you finish, print the file tree, the section 20 checklist with every line marked, the base selection table, the champion twin confirmed accuracy on readback test with its interval beside the base and the floor, the methods table at one ceiling, pass at 1 and pass at 16 before and after, every defect arm and control with its prediction, outcome and firing step, the mutation score per family, the contamination counts, the audit result, the judge calibration, the regression suite, Spider, the sizing totals, the parity result, the latency with the cold start, the live URLs, and anything you could not measure or any dependency that was missing today.

