"use client";

// The methods on one budget: each method's twin confirmed accuracy on a held-out set from the graded
// completions, its interval and the champion from results/manifest.json, and the GPU hours its runs
// used from the run ledger. The manifest keys read here are listed in web/README.md.
import { loadIndex, loadRunLines, loadSource, type DataIndex } from "./data";
import { setScores, type SetScore } from "./grades";
import { ManifestView, type Manifest } from "./manifest";
import type { Difference } from "./messages";
import { FLOOR, METHODS, methodRuns, type MethodCode } from "./runs";
import type { LedgerRow } from "./types";

export interface MethodScore {
  code: string;
  label: string;
  token: string;
  runs: string[];
  score: SetScore | null;
  low: number | null;
  high: number | null;
  gpuHours: number | null;
}

/** Ledger stages that count toward a method's ceiling: training and the sampling it needs, not evaluation. */
export const TRAINING_STAGES = new Set(["sft", "dpo", "pairs", "grpo", "select"]);

export async function loadManifest(): Promise<ManifestView> {
  return new ManifestView(await loadSource<Manifest>("results/manifest.json"));
}

/** An interval from the manifest table methods.accuracy, if the repository has computed it. */
export function intervalFor(manifest: ManifestView, method: string, set: string, metric: string): { low: number; high: number } | null {
  const row = manifest
    .rows("methods.accuracy")
    .find((r) => String(r.method) === method && String(r.set) === set && String(r.metric) === metric);
  if (!row || typeof row.low !== "number" || typeof row.high !== "number") return null;
  return { low: row.low, high: row.high };
}

/** A paired difference a minus b from the manifest table methods.differences. */
export function differenceFor(manifest: ManifestView, a: string, b: string, set: string, metric: string): Difference | null {
  const row = manifest
    .rows("methods.differences")
    .find((r) => String(r.a) === a && String(r.b) === b && String(r.set) === set && String(r.metric) === metric);
  if (!row || typeof row.point !== "number" || typeof row.low !== "number" || typeof row.high !== "number") return null;
  return { point: row.point, low: row.low, high: row.high, clusters: typeof row.clusters === "number" ? row.clusters : 0 };
}

export function championOf(manifest: ManifestView): string | null {
  const v = manifest.value("methods.champion") ?? manifest.value("models.champion");
  return typeof v === "string" && v.trim() ? v.trim() : null;
}

async function gpuHoursOf(index: DataIndex, runs: string[]): Promise<number | null> {
  let seconds = 0;
  let seen = false;
  for (const id of runs) {
    const entry = index.runs[id];
    if (!entry) continue;
    for (const row of await loadRunLines<LedgerRow>(entry, "ledger")) {
      if (!TRAINING_STAGES.has(row.stage)) continue;
      seen = true;
      seconds += Object.values(row.gpu_seconds ?? {}).reduce((s, v) => s + v, 0);
    }
  }
  return seen ? seconds / 3600 : null;
}

/** The greedy row when there is one, else the sampled row; the latest run of the method. */
function pick(rows: SetScore[], runs: string[]): SetScore | null {
  for (const run of [...runs].reverse()) {
    const mine = rows.filter((r) => r.request === run);
    const greedy = mine.find((r) => Number(r.temperature) === 0);
    if (greedy) return greedy;
    if (mine[0]) return mine[0];
  }
  return null;
}

export interface MethodsData {
  index: DataIndex;
  manifest: ManifestView;
  set: string;
  methods: MethodScore[];
  floor: MethodScore;
  champion: string | null;
  sql: string;
  graded: boolean;
}

const methodsOnce = new Map<string, Promise<MethodsData>>();

/** The methods on one set, computed once per visit however many sections ask. */
export function loadMethods(set: string): Promise<MethodsData> {
  let hit = methodsOnce.get(set);
  if (!hit) {
    hit = computeMethods(set).catch((err: unknown) => {
      methodsOnce.delete(set);
      throw err;
    });
    methodsOnce.set(set, hit);
  }
  return hit;
}

async function computeMethods(set: string): Promise<MethodsData> {
  const [index, manifest] = await Promise.all([loadIndex(), loadManifest()]);
  const byMethod = methodRuns(index);
  const requests = [...Object.values(byMethod).flat(), FLOOR];
  const graded = Object.values(index.grades).some((g) => g.set === set);
  const scores = graded ? await setScores(index, [set], requests) : { sql: "", rows: [] };
  const methods: MethodScore[] = [];
  for (const m of METHODS) {
    const runs = byMethod[m.code as MethodCode] ?? [];
    const score = pick(scores.rows, runs);
    const interval = intervalFor(manifest, m.code, set, "twin_confirmed_accuracy");
    methods.push({ code: m.code, label: `${m.code} ${m.name}`, token: m.token, runs, score, low: interval?.low ?? null, high: interval?.high ?? null, gpuHours: await gpuHoursOf(index, runs) });
  }
  const floorScore = pick(scores.rows, [FLOOR]);
  const floorInterval = intervalFor(manifest, "floor", set, "twin_confirmed_accuracy");
  return {
    index,
    manifest,
    set,
    methods,
    floor: { code: "floor", label: "the floor", token: "control", runs: [FLOOR], score: floorScore, low: floorInterval?.low ?? null, high: floorInterval?.high ?? null, gpuHours: null },
    champion: championOf(manifest),
    sql: scores.sql,
    graded,
  };
}
