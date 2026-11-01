// The rows behind every chart drawn from a JSON file, written twice: as the DuckDB SQL printed by the
// chart's "show the SQL" toggle (run it from the repository root with the duckdb CLI or Python
// package), and as the function the page runs. tests/unit/jsonsql.test.ts runs each query on the
// repository's own files with DuckDB and checks it returns the rows the function returns.
import type { Contamination, Mutation, Plan, QuestionsManifest, TwinsLock } from "./types";

export type Row = Record<string, string | number | null>;

export interface JsonQuery<T> {
  source: string;
  sql: string;
  rows: (json: T) => Row[];
}

const share = (n: number, d: number) => (d ? n / d : null);
const familyOrder = (a: Row, b: Row) => Number(a.family === "all") - Number(b.family === "all") || String(a.family).localeCompare(String(b.family));

export const mutationByFamily: JsonQuery<Mutation> = {
  source: "results/mutation.json",
  sql: `-- The mutation score on the test golds: of the near misses that are not equivalent to the gold,
-- the share each reward pays (the honest one in full) and the share the twins confirm.
SELECT family, mutants, equivalent, scored, honest_paid_full, lenient_paid, twin_confirmed,
       honest_paid_full / scored AS honest_pays,
       lenient_paid / scored AS lenient_pays,
       twin_confirmed / scored AS twins_confirm
FROM (SELECT unnest(test, recursive := true) FROM read_json_auto('results/mutation.json'))
ORDER BY family = 'all', family`,
  rows: (m) =>
    m.test
      .map((r) => ({
        family: r.family,
        mutants: r.mutants,
        equivalent: r.equivalent,
        scored: r.scored,
        honest_paid_full: r.honest_paid_full,
        lenient_paid: r.lenient_paid,
        twin_confirmed: r.twin_confirmed,
        honest_pays: share(r.honest_paid_full, r.scored),
        lenient_pays: share(r.lenient_paid, r.scored),
        twins_confirm: share(r.twin_confirmed, r.scored),
      }))
      .sort(familyOrder),
};

export const distillation: JsonQuery<TwinsLock> = {
  source: "data/twins.lock.json",
  sql: `-- The greedy cover: the twins in the order they were kept, what each added, and the running total
-- of train and dev mutants the original pays and the kept twins catch.
WITH t AS (SELECT kept, distillation FROM read_json_auto('data/twins.lock.json')),
     ranked AS (SELECT unnest(range(1, len(kept) + 1)) AS rank, kept, distillation FROM t)
SELECT rank, kept[rank] AS twin, distillation.gains[rank] AS caught,
       sum(distillation.gains[rank]) OVER (ORDER BY rank) AS caught_so_far,
       distillation.original_passing AS original_passing
FROM ranked
ORDER BY rank`,
  rows: (lock) => {
    let total = 0;
    return lock.kept.map((twin, i) => {
      const caught = lock.distillation.gains[i] ?? 0;
      total += caught;
      return { rank: i + 1, twin, caught, caught_so_far: total, original_passing: lock.distillation.original_passing };
    });
  },
};

export const twinAlone: JsonQuery<TwinsLock> = {
  source: "data/twins.lock.json",
  sql: `-- Each of the candidate twins alone: how many of the mutants the original pays it catches.
UNPIVOT (SELECT c.* FROM (SELECT distillation.caught_by_twin AS c FROM read_json_auto('data/twins.lock.json')))
ON COLUMNS(*) INTO NAME twin VALUE caught
ORDER BY caught DESC, twin`,
  rows: (lock) =>
    Object.entries(lock.distillation.caught_by_twin)
      .map(([twin, caught]) => ({ twin, caught }))
      .sort((a, b) => b.caught - a.caught || a.twin.localeCompare(b.twin)),
};

export const drops: JsonQuery<QuestionsManifest> = {
  source: "data/questions/manifest.json",
  sql: `-- Every candidate question dropped, by the rule that dropped it.
UNPIVOT (SELECT drops.* FROM read_json_auto('data/questions/manifest.json'))
ON COLUMNS(*) INTO NAME reason VALUE dropped
ORDER BY dropped DESC, reason`,
  rows: (m) =>
    Object.entries(m.drops)
      .map(([reason, dropped]) => ({ reason, dropped }))
      .sort((a, b) => b.dropped - a.dropped || a.reason.localeCompare(b.reason)),
};

export const sets: JsonQuery<QuestionsManifest> = {
  source: "data/questions/manifest.json",
  sql: `-- The size of every set the generator wrote, with the share of its gold tables the schema search returns.
WITH s AS (UNPIVOT (SELECT sets.* FROM read_json_auto('data/questions/manifest.json')) ON COLUMNS(*) INTO NAME "set" VALUE questions),
     r AS (UNPIVOT (SELECT schema_recall.* FROM read_json_auto('data/questions/manifest.json')) ON COLUMNS(*) INTO NAME "set" VALUE schema_recall)
SELECT s."set", s.questions, r.schema_recall
FROM s LEFT JOIN r USING ("set")
ORDER BY s.questions DESC, s."set"`,
  rows: (m) =>
    Object.entries(m.sets)
      .map(([set, questions]) => ({ set, questions, schema_recall: m.schema_recall[set] ?? null }))
      .sort((a, b) => b.questions - a.questions || a.set.localeCompare(b.set)),
};

export const contaminationChecks: JsonQuery<Contamination> = {
  source: "results/contamination.json",
  sql: `-- What the contamination report found between the held-out sets and training and development.
UNPIVOT (SELECT exact, tuple_overlap, ngram_overlap, embedding_duplicates, near_duplicates,
                len(moved) AS moved
         FROM read_json_auto('results/contamination.json'))
ON COLUMNS(*) INTO NAME check_name VALUE found
ORDER BY found DESC, check_name`,
  rows: (c) =>
    (
      [
        ["exact", c.exact],
        ["tuple_overlap", c.tuple_overlap],
        ["ngram_overlap", c.ngram_overlap],
        ["embedding_duplicates", c.embedding_duplicates],
        ["near_duplicates", c.near_duplicates],
        ["moved", c.moved.length],
      ] as [string, number][]
    )
      .map(([check_name, found]) => ({ check_name, found }))
      .sort((a, b) => b.found - a.found || a.check_name.localeCompare(b.check_name)),
};

export const quotaPlan: JsonQuery<Plan> = {
  source: "evaluation/plan.yaml",
  sql: `-- The registered quota plan, stage by stage, in GPU hours (evaluation/plan.yaml as JSON).
UNPIVOT (SELECT q.* EXCLUDE (mandatory_total) FROM (SELECT compute.quota_plan AS q FROM read_json_auto('evaluation/plan.json')))
ON COLUMNS(*) INTO NAME stage VALUE gpu_hours
ORDER BY gpu_hours DESC, stage`,
  rows: (p) =>
    Object.entries(p.compute?.quota_plan ?? {})
      .filter(([stage]) => stage !== "mandatory_total")
      .map(([stage, gpu_hours]) => ({ stage, gpu_hours }))
      .sort((a, b) => b.gpu_hours - a.gpu_hours || a.stage.localeCompare(b.stage)),
};
