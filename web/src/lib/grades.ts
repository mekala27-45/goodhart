"use client";

// The questions the pages ask of the graded completions, as the SQL DuckDB-WASM runs. Every metric
// follows packages/evaluation/src/goodhart_evaluation/metrics.py: an item's value is the share of its
// samples that pass, and a set's value is the mean over items, so a question with sixteen samples
// counts once. Twin confirmed accuracy is over answerable items only. The SQL printed beside a
// chart is the string defined here.
import type { DataIndex, GradeEntry } from "./data";
import { queryGrades } from "./duck";
import { meanPassAtK, KS } from "./passk";

export const lit = (s: string) => `'${s.replace(/'/g, "''")}'`;
const inList = (xs: string[]) => (xs.length ? xs.map(lit).join(", ") : "NULL");

/** The grade files of the given runs and sets. */
export function gradeFiles(index: DataIndex, runs: string[] | null, sets: string[] | null): GradeEntry[] {
  return Object.values(index.grades).filter((g) => (!runs || runs.includes(g.run)) && (!sets || sets.includes(g.set)));
}

export function devCurveSql(runs: string[]): string {
  return `-- Twin confirmed accuracy on dev at every saved checkpoint: the share of answerable dev
-- questions whose greedy answer matches the gold on the original and on every kept twin.
SELECT request AS run,
       CAST(regexp_extract(checkpoint, 'step-([0-9]+)', 1) AS INTEGER) AS step,
       avg(confirmed_share) AS twin_confirmed,
       avg(original_share) AS execution,
       count(*) AS items
FROM (
  SELECT request, checkpoint, item_id,
         avg(CASE WHEN confirmed THEN 1.0 ELSE 0.0 END) AS confirmed_share,
         avg(CASE WHEN original THEN 1.0 ELSE 0.0 END) AS original_share
  FROM grades
  WHERE "set" = 'dev' AND NOT refuse_item
    AND regexp_matches(checkpoint, 'step-[0-9]+')
    AND request IN (${inList(runs)})
  GROUP BY request, checkpoint, item_id
)
GROUP BY run, step
ORDER BY run, step`;
}

export interface CurvePoint {
  run: string;
  step: number;
  twin_confirmed: number;
  execution: number;
  items: number;
}

export async function devCurves(index: DataIndex, runs: string[]): Promise<{ sql: string; rows: CurvePoint[] }> {
  const sql = devCurveSql(runs);
  const files = gradeFiles(index, runs, ["dev"]);
  if (!files.length) return { sql, rows: [] };
  return { sql, rows: await queryGrades<CurvePoint & Record<string, unknown>>(sql, files) };
}

export function setScoresSql(sets: string[], requests: string[] | null): string {
  const who = requests ? `\n    AND request IN (${inList(requests)})` : "";
  return `-- Per checkpoint and temperature: each item's share of passing samples, then the mean over items.
WITH per_item AS (
  SELECT request, checkpoint, "set", temperature, item_id, refuse_item,
         count(*) AS n,
         sum(CASE WHEN confirmed THEN 1 ELSE 0 END) AS c,
         avg(CASE WHEN original THEN 1.0 ELSE 0.0 END) AS original_share,
         avg(CASE WHEN confirmed THEN 1.0 ELSE 0.0 END) AS confirmed_share,
         avg(CASE WHEN refused THEN 1.0 ELSE 0.0 END) AS refused_share,
         avg(tokens) AS tokens
  FROM grades
  WHERE "set" IN (${inList(sets)})${who}
  GROUP BY request, checkpoint, "set", temperature, item_id, refuse_item
)
SELECT request, checkpoint, "set", temperature,
       avg(confirmed_share) FILTER (WHERE NOT refuse_item) AS twin_confirmed,
       avg(original_share) FILTER (WHERE NOT refuse_item) AS execution,
       avg(refused_share) FILTER (WHERE refuse_item) AS refusal_accuracy,
       avg(refused_share) FILTER (WHERE NOT refuse_item) AS over_refusal,
       count(*) FILTER (WHERE NOT refuse_item) AS answerable,
       max(n) AS samples,
       avg(tokens) AS tokens
FROM per_item
GROUP BY request, checkpoint, "set", temperature
ORDER BY request, checkpoint, "set", temperature`;
}

export interface SetScore {
  request: string;
  checkpoint: string;
  set: string;
  temperature: number;
  twin_confirmed: number | null;
  execution: number | null;
  refusal_accuracy: number | null;
  over_refusal: number | null;
  answerable: number;
  samples: number;
  tokens: number | null;
}

export async function setScores(index: DataIndex, sets: string[], requests: string[] | null): Promise<{ sql: string; rows: SetScore[] }> {
  const sql = setScoresSql(sets, requests);
  const files = Object.values(index.grades).filter((g) => sets.includes(g.set));
  if (!files.length) return { sql, rows: [] };
  return { sql, rows: await queryGrades<SetScore & Record<string, unknown>>(sql, files) };
}

export function itemCountsSql(set: string, requests: string[]): string {
  return `-- Samples and twin confirmed samples per answerable item, for the unbiased pass at k.
SELECT request, checkpoint, item_id,
       count(*) AS n,
       sum(CASE WHEN confirmed THEN 1 ELSE 0 END) AS c
FROM grades
WHERE "set" = ${lit(set)} AND NOT refuse_item AND temperature > 0
  AND request IN (${inList(requests)})
GROUP BY request, checkpoint, item_id
ORDER BY request, checkpoint, item_id`;
}

export interface Coverage {
  request: string;
  checkpoint: string;
  points: { k: number; value: number | null; items: number }[];
}

/** pass at k for k in the registered list, per request and checkpoint, from sampled completions. */
export async function coverage(index: DataIndex, set: string, requests: string[]): Promise<{ sql: string; rows: Coverage[] }> {
  const sql = itemCountsSql(set, requests);
  const files = gradeFiles(index, null, [set]);
  if (!files.length || !requests.length) return { sql, rows: [] };
  const rows = await queryGrades<{ request: string; checkpoint: string; item_id: string; n: number; c: number }>(sql, files);
  const groups = new Map<string, { request: string; checkpoint: string; items: { n: number; c: number }[] }>();
  for (const r of rows) {
    const key = `${r.request}|${r.checkpoint}`;
    const g = groups.get(key) ?? { request: r.request, checkpoint: r.checkpoint, items: [] };
    g.items.push({ n: Number(r.n), c: Number(r.c) });
    groups.set(key, g);
  }
  return {
    sql,
    rows: [...groups.values()].map((g) => ({
      request: g.request,
      checkpoint: g.checkpoint,
      points: KS.map((k) => ({ k, ...meanPassAtK(g.items, k) })),
    })),
  };
}

export function hackRatesSql(runs: string[], set: string): string {
  return `-- The share of completions each hack detector flags, per run and checkpoint.
SELECT request AS run, checkpoint,
       avg(CASE WHEN flag_literal_only THEN 1.0 ELSE 0.0 END) AS literal_only,
       avg(CASE WHEN flag_copied_value THEN 1.0 ELSE 0.0 END) AS copied_value,
       avg(CASE WHEN flag_unfiltered THEN 1.0 ELSE 0.0 END) AS unfiltered,
       avg(CASE WHEN flag_refusal_answerable THEN 1.0 ELSE 0.0 END) AS refusal_answerable,
       avg(CASE WHEN flag_valid_but_wrong THEN 1.0 ELSE 0.0 END) AS valid_but_wrong,
       count(*) AS completions
FROM grades
WHERE "set" = ${lit(set)} AND request IN (${inList(runs)})
GROUP BY request, checkpoint
ORDER BY request, checkpoint`;
}

export function marksSql(set: string, items: string[], requests: string[]): string {
  return `-- The repository's grade of each script shown: one row per completion.
SELECT item_id, request, checkpoint, sample, temperature, refuse_item, kind, executed, original,
       confirmed, refused, tokens, flag_literal_only, flag_copied_value, flag_unfiltered,
       flag_refusal_answerable, flag_valid_but_wrong
FROM grades
WHERE "set" = ${lit(set)} AND item_id IN (${inList(items)}) AND request IN (${inList(requests)})
ORDER BY item_id, request, checkpoint, temperature, sample`;
}

export interface MarkRow {
  item_id: string;
  request: string;
  checkpoint: string;
  sample: number;
  temperature: number;
  refuse_item: boolean;
  kind: string;
  executed: boolean;
  original: boolean;
  confirmed: boolean;
  refused: boolean;
  tokens: number;
  flag_literal_only: boolean;
  flag_copied_value: boolean;
  flag_unfiltered: boolean;
  flag_refusal_answerable: boolean;
  flag_valid_but_wrong: boolean;
  [key: string]: unknown;
}

export async function marks(index: DataIndex, set: string, items: string[], requests: string[]): Promise<{ sql: string; rows: MarkRow[] }> {
  const sql = marksSql(set, items, requests);
  const files = gradeFiles(index, requests, [set]);
  if (!files.length || !items.length) return { sql, rows: [] };
  return { sql, rows: await queryGrades<MarkRow>(sql, files) };
}

export function flaggedSql(run: string, checkpoint: string, flag: string): string {
  return `-- Dev answers at the last checkpoint, the ones the predicted detector flagged first.
SELECT *, ${flag} AS flagged
FROM grades
WHERE "set" = 'dev' AND request = ${lit(run)} AND checkpoint = ${lit(checkpoint)}
ORDER BY ${flag} DESC, original DESC, confirmed ASC, item_id
LIMIT 3`;
}
