"use client";

// DuckDB-WASM over the graded completions. The engine starts in a worker the first time a page asks
// a question of the grades, never on a page that does not: most pages read JSON and never load it.
// Each results/grades/<run>/<file>.parquet is fetched once, decoded (lib/parquet.ts) and inserted as
// its own table; the view `grades` is every loaded file, so the SQL beside a chart reads `grades`.
import type { AsyncDuckDB, AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";

import type { GradeEntry } from "./data";
import { getBytes } from "./fetcher";
import { decodeParquet, toArrowStream } from "./parquet";
import { BASE, dataUrl } from "./site";

export type Row = Record<string, unknown>;

let engine: Promise<{ db: AsyncDuckDB; conn: AsyncDuckDBConnection }> | null = null;
const tables = new Map<string, string>();
let lock: Promise<unknown> = Promise.resolve();

async function boot() {
  const duckdb = await import("@duckdb/duckdb-wasm/dist/duckdb-browser.mjs");
  const worker = new Worker(`${BASE}/duckdb/duckdb-browser-eh.worker.js`);
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(new URL(`${BASE}/duckdb/duckdb-eh.wasm`, window.location.href).href);
  // Counts arrive as doubles: every count here is far below 2^53, and the charts want numbers.
  await db.open({ query: { castBigIntToDouble: true } });
  const conn = await db.connect();
  for (const setting of ["SET autoinstall_known_extensions = false", "SET autoload_known_extensions = false"]) {
    await conn.query(setting).catch(() => undefined);
  }
  return { db, conn };
}

function serial<T>(work: () => Promise<T>): Promise<T> {
  const next = lock.then(work, work);
  lock = next.catch(() => undefined);
  return next;
}

/** Load the named grade files (once each) and point the view `grades` at every file loaded so far. */
export function ensureGrades(entries: GradeEntry[]): Promise<void> {
  return serial(async () => {
    engine ??= boot();
    const { conn } = await engine;
    let added = false;
    for (const entry of entries) {
      if (tables.has(entry.url)) continue;
      const name = `g_${tables.size}`;
      const decoded = await decodeParquet(await getBytes(dataUrl(entry.url)));
      await conn.insertArrowFromIPCStream(await toArrowStream(decoded), { name, create: true });
      tables.set(entry.url, name);
      added = true;
    }
    if (added || tables.size === 0) {
      const names = [...tables.values()];
      const body = names.length
        ? names.map((n) => `SELECT * FROM ${n}`).join(" UNION ALL BY NAME ")
        : "SELECT NULL::VARCHAR AS item_id WHERE FALSE";
      await conn.query(`CREATE OR REPLACE VIEW grades AS ${body}`);
    }
  });
}

/** Run SQL over the view `grades` after loading the files it needs. Rows come back as plain objects. */
export async function queryGrades<T extends Row = Row>(sql: string, entries: GradeEntry[]): Promise<T[]> {
  await ensureGrades(entries);
  const { conn } = await engine!;
  const table = await serial(() => conn.query(sql));
  const fields = table.schema.fields.map((f) => f.name);
  return table.toArray().map((r) => {
    const json = r.toJSON() as Row;
    const out: Row = {};
    for (const f of fields) {
      const v = json[f];
      out[f] = typeof v === "bigint" ? Number(v) : v;
    }
    return out as T;
  });
}
