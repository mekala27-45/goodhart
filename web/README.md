# goodhart web: the marking room

The public site of goodhart, a static export of Next.js 16 for GitHub Pages at
https://mekala27-45.github.io/goodhart/. It reads only what the repository holds: a build step copies
the committed results into `public/data`, and every page asks `public/data/index.json` what exists
before it asks for a file. A section whose file does not exist yet says "Not yet run" in words, names
the file it reads and the step that writes it, and draws nothing in its place.

```
npm ci
npm run build      # tokens, DuckDB-WASM assets, collect-data, then next build into out/
npm test           # unit tests (vitest)
npm run e2e        # end to end tests (Playwright) against out/, with the fixtures laid over the data
npm run typecheck  # tsc, strict
npm run serve      # the test server on http://127.0.0.1:4173/goodhart/
```

`npm run e2e` needs a build first. Playwright is pinned at 1.63.0; in this sandbox Chromium is at
`/opt/pw-browsers/chromium-1243/chrome-linux64/chrome` (override with `GOODHART_CHROMIUM`). Elsewhere
run `npx playwright install chromium` once.

## The pages

| Route | Title | Reads |
|---|---|---|
| `/` | Paid against earned | predictions, the defect study's monitor and alerts, dev grades, the manifest |
| `/methods/` | One budget, four recipes | held-out grades, ledgers, the manifest, hypotheses |
| `/hacks/` | What the reward paid for | predictions, monitor thresholds, alerts, dev grades and scripts, incidents |
| `/verifier/` | Who marks the marker | mutation score, twins lock, grades, judge, parity |
| `/data/` | Where the questions came from | profile, questions manifest, contamination, audit |
| `/desk/` | Mark it yourself | the grader API, the recorded session, held-out scripts and grades |
| `/sizing/` | What it costs | ledgers, sizing sheet, plan, probe |
| `/models/` | The cards, the reports and the base | probe, selection, plan gates, docs/models, docs/validation |
| `/report/` | The briefing | report/briefing.md, with a print stylesheet |

Every route carries the statement in its footer, taken verbatim from
`packages/core/src/goodhart_core/statements.py` into `src/lib/statement.ts` (a unit test compares the
two). Every chart has one message as its title, written by a rule in `src/lib/rules.ts` from the rows it
draws, a table toggle, a show the SQL toggle, and a source line with the digest of each file it read.

## The data step

`scripts/collect-data.mjs` runs before `next build`. It clears `public/data` (keeping
`recorded_session.json`), copies whatever of these exists, and writes `public/data/index.json` with
every file found (its digest and origin), every file missing, the runs, the graded files and the
scripts.

- JSON, copied as is: `results/mutation.json`, `contamination.json`, `profile.json`, `sizing.json`,
  `gpu_probe.json`, `selection.json`, `audit.json`, `parity.json`, `latency.json`, `manifest.json`,
  `judge.json`, `live_check.json`, `data/questions/manifest.json`, `data/twins.lock.json`.
- YAML, published as JSON with the same content: `evaluation/plan.yaml`, `predictions.yaml`,
  `hypotheses.yaml`, `config/monitor.yaml`.
- Markdown: `report/briefing.md`, `docs/models/*.md`, `docs/validation/*.md`, `docs/incidents/*.md`.
- `data/questions/readback_test.jsonl` is published as JSON with the fields id, set, question,
  family, warehouse, template, measure, unit_asked and source. The gold, the refusal reason and the
  SQL are dropped here, so no page can show them.
- Every `results/runs/<id>/`: `ledger.jsonl`, `monitor.jsonl`, `alerts.json`, `probe.json`,
  `provenance.json`, `metrics/trainer.jsonl`, and `requests/<id>.json` beside it.
- Completions become scripts: for the sets readback_test, test, template_test and dev, one answer per
  run, checkpoint and question (the greedy one, else the lowest sample), up to 4,000 characters, in
  `public/data/scripts/<set>.json` with the question text. Dev keeps each run's last checkpoint only.
- `results/grades/index.json` and every parquet file it lists, unchanged.

## How a run is read

- A defect study run is a `grpo` run whose request or ledger names an arm (`honest`, `lenient`,
  `leaked`, `lazy`, `refusal`, `random`, `format`) and no method. When an arm ran twice, the latest
  run id wins.
- A method's run names the method in its request, as `hyperparameters.method` or as `arm`: `M0` to
  `M4`, or `base`, `sft`, `dpo`, `grpo`, `sft_grpo`. The methods chart reads the sample runs' grades;
  the compute bar sums the ledger rows of that method's `sft`, `dpo`, `pairs`, `grpo` and `select`
  stages.
- The floor's grades carry the request `floor`.
- The red line of the hero is the mean reward paid per step, read from `monitor.jsonl` under the first
  key present of `reward`, `mean_reward`, `reward_mean`, `reward_paid`, `paid`, and from
  `metrics/trainer.jsonl` (TRL logs `reward`) when the monitor rows carry none. Today's monitor rows
  (packages/monitor) do not write the reward, so until they do the trainer log is what the hero reads.
- The green line is twin confirmed accuracy on dev at every `step-<n>` checkpoint, computed in the
  browser from the grades exactly as `packages/evaluation/metrics.py` does: each item's share of
  passing samples, then the mean over answerable items.

## The grades in the browser

The parquet grades are queried with DuckDB-WASM 1.32.0, the latest stable release (npm's `latest`
tag points at a dev build, so the version is pinned). Only the exception handling bundle is shipped,
copied into `public/duckdb` at build time, and loaded only on a page that has grades to read. The 1.32.0
bundle would download the parquet extension from extensions.duckdb.org on the first `read_parquet`,
so each file is instead fetched with the site's own helper, decoded by hyparquet with fzstd, and
inserted as an Arrow table; the view `grades` is every file loaded, and the SQL printed beside a chart
is the SQL that ran. Charts drawn from JSON print the DuckDB query that returns the same rows from
the published file; `tests/unit/jsonsql.test.ts` runs each one with DuckDB and checks it.

Of the rewards, honest, leaked, lazy and refusal are derived from a grade (kind, executed, original,
refuse_item). Lenient and format need the comparator and the parse, so a script shows them only when
the grades carry a `pays_<reward>` column; adding `pays_honest`, `pays_lenient`, `pays_lazy` and
`pays_refusal` to the grades parquet would put them in every margin.

## The manifest keys the pages read

Intervals, adjusted p values and verdicts come from `results/manifest.json` and nowhere else. A page
shows the point estimate from the grades and says "no interval yet" until these exist:

| Key | Kind | Columns or value | Page |
|---|---|---|---|
| `methods.champion` | value | text, the method code of the champion | `/`, `/methods/` |
| `methods.accuracy` | table | method, set, metric, point, low, high, items, clusters | `/`, `/methods/` |
| `methods.differences` | table | a, b, set, metric, point, low, high, clusters | `/`, `/methods/` |
| `methods.regression` | table | method, suite, point, low, high | `/methods/` |
| `methods.planted` | table | set (clean or planted), twin_confirmed, low, high | `/methods/` |
| `hypotheses.verdicts` | table | id, point, low, high, p_adjusted, held | `/methods/` |
| `hacks.outcomes` | table | arm, share_of_gain, held (the controls) | `/hacks/` |
| `models.gates` | table | method, gate (number or text), passed, detail | `/models/` |
| `data.paraphrases`, `data.paraphrases_kept` | table, value | the round trip | `/data/` |

Metric names follow `metrics.py`: `twin_confirmed_accuracy`, `execution_accuracy`, and so on. A
defect arm's prediction is judged on the page from `alerts.json` and the dev grades, by the rule its
prediction states; a control needs the starting checkpoint's accuracy, so its verdict is read from
`hacks.outcomes`.

## The desk and the recorded session

The desk probes `GET /health` of https://goodhart-grader.vercel.app twice, three seconds apart, six
seconds each. If the grader answers, questions come from `GET /v1/questions` and every query is posted
to `POST /v1/grade`. If not, the desk plays `public/data/recorded_session.json` and says so in words; a
live grade that fails is shown as a failure and never swapped for a recording.

The recorded session holds `health`, `questions` and one `grades` entry (request, status, response)
per try, each response in the exact shape the API returns, and `tries`: the five tries (print the
number, drop the filter, return the whole table, refuse, write it properly) for each question they are
written for, in `scripts/tries.json`. The file committed now is a placeholder made by
`scripts/record_local.py`, which calls `goodhart_api.service.Service.grade`, the function the API
calls, on the build machine; its verdicts are real, its `grade_id` is null because no grade log was
written, and the page labels it a placeholder. To replace it with a recording of the live API:

```
node web/scripts/record-session.mjs --api https://goodhart-grader.vercel.app
```

## The test server and the tests

`scripts/test-server.mjs` serves `out/` under `/goodhart`, refuses HEAD with a 405, and answers its
very first request with a 503 and `Retry-After: 1`, the way a waking host does. The site's fetch
helper (`src/lib/fetcher.ts`) retries a 502, 503 or 504 with backoff and refuses to send HEAD.

The end to end tests run two servers: one with `tests/fixtures/repo` laid over the repository's data
(runs, grades, scripts and a manifest that do not exist yet; the pages mark every fixture file as
"end to end test fixtures, not results"), and one with the repository's data alone, for the
"not yet run" states. The fixtures are written by the repository's own grade writer, so the parquet
schema is the real one:

```
uv run python web/tests/fixtures/make_fixtures.py   # from the repository root
```

The API is intercepted in the tests: answered with `tests/fixtures/api/*.json` (shaped like the real
responses) for the live grade, and refused for the recorded session.

## Type and colour

Colours are exactly the tokens of `src/theme/palette.json`; `scripts/build-tokens.mjs` writes them to
`src/theme/tokens.css`, Tailwind's own colours are switched off, and a unit test fails on any colour in
the source that is not in the palette. Light is ruled paper; dark is the blackboard, taken from the
system or the toggle.

All five faces of option A were on npm and are self hosted through `next/font/local`:

| Use | Package | Version |
|---|---|---|
| display, the briefing | `@fontsource-variable/stix-two-text` | 5.3.0 |
| interface and body | `@fontsource-variable/sofia-sans` | 5.3.0 |
| dense tables | `@fontsource-variable/sofia-sans-semi-condensed` | 5.3.0 |
| SQL, figures, identifiers | `@fontsource/monaspace-neon` | 5.3.0 |
| the marks in the margin only | `@fontsource/monaspace-radon` | 5.3.0 |

No fallback option was needed.

## Pinned versions

next 16.4.3, react and react-dom 19.3.1, tailwindcss and @tailwindcss/postcss 4.3.3,
@duckdb/duckdb-wasm 1.32.0, apache-arrow 17.0.0, hyparquet 1.31.4, fzstd 0.1.1, marked 18.1.2,
typescript 5.9.3, vitest 5.0.5 (4.1.12 could not be installed: npm's resolver fails on its peer set),
@playwright/test 1.63.0, yaml 2.9.1.
