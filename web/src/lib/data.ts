// The published data, as scripts/collect-data.mjs wrote it: public/data/index.json says what exists,
// and every page asks the index before it asks for a file, so a missing file is a "not yet run"
// section and never a failed request.
import { getJson, getText } from "./fetcher";
import { dataUrl } from "./site";

export type Origin = "repository" | "fixture";

export interface SourceEntry {
  url: string;
  bytes: number;
  sha256: string;
  origin: Origin;
}

export type RunFile = "ledger" | "monitor" | "alerts" | "probe" | "provenance" | "trainer" | "request";

export interface RunEntry {
  stage?: string | null;
  model?: string | null;
  arm?: string | null;
  reward?: string | null;
  seed?: number | null;
  method?: string | null;
  status?: string | null;
  steps?: number | null;
  files: Partial<Record<RunFile, string>>;
  completions: string[];
}

export interface GradeEntry {
  url: string;
  run: string;
  name: string;
  set: string;
  rows: number | null;
  bytes: number;
  origin: Origin;
}

export interface ScriptsEntry {
  url: string;
  items: number;
  runs: string[];
  bytes: number;
}

export interface DocEntry {
  name: string;
  url: string;
}

export interface DataIndex {
  schema: number;
  collected_at: string;
  sources: Record<string, SourceEntry>;
  missing: string[];
  runs: Record<string, RunEntry>;
  grades: Record<string, GradeEntry>;
  scripts: Record<string, ScriptsEntry>;
  docs: { models: DocEntry[]; validation: DocEntry[]; incidents: DocEntry[] };
}

let indexPromise: Promise<DataIndex> | null = null;

/** The index of what the build found, fetched once per visit. */
export function loadIndex(): Promise<DataIndex> {
  indexPromise ??= getJson<DataIndex>(dataUrl("index.json")).catch((err: unknown) => {
    indexPromise = null;
    throw err;
  });
  return indexPromise;
}

const cache = new Map<string, Promise<unknown>>();

function once<T>(key: string, load: () => Promise<T>): Promise<T> {
  let hit = cache.get(key) as Promise<T> | undefined;
  if (!hit) {
    hit = load().catch((err: unknown) => {
      cache.delete(key);
      throw err;
    });
    cache.set(key, hit);
  }
  return hit;
}

/** A source by its repository path, parsed as JSON; null when the repository does not hold it yet. */
export async function loadSource<T>(source: string): Promise<T | null> {
  const index = await loadIndex();
  const entry = index.sources[source];
  if (!entry) return null;
  return once(`json:${entry.url}`, () => getJson<T>(dataUrl(entry.url)));
}

/** A source as text (markdown, JSON lines). */
export async function loadSourceText(source: string): Promise<string | null> {
  const index = await loadIndex();
  const entry = index.sources[source];
  if (!entry) return null;
  return once(`text:${entry.url}`, () => getText(dataUrl(entry.url)));
}

/** A published file by its URL under public/data, as JSON. */
export function loadUrl<T>(url: string): Promise<T> {
  return once(`json:${url}`, () => getJson<T>(dataUrl(url)));
}

export function loadUrlText(url: string): Promise<string> {
  return once(`text:${url}`, () => getText(dataUrl(url)));
}

export function parseJsonLines<T>(text: string): T[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

/** A run's JSON lines file (ledger, monitor, trainer), or an empty list when the run has none. */
export async function loadRunLines<T>(run: RunEntry, file: RunFile): Promise<T[]> {
  const url = run.files[file];
  if (!url) return [];
  return parseJsonLines<T>(await loadUrlText(url));
}

export async function loadRunJson<T>(run: RunEntry, file: RunFile): Promise<T | null> {
  const url = run.files[file];
  return url ? loadUrl<T>(url) : null;
}

/** Whether a source came from the end to end fixtures rather than the repository. */
export function isFixture(index: DataIndex, source: string): boolean {
  return index.sources[source]?.origin === "fixture";
}
