import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import * as queries from "@/lib/jsonsql";

// @ts-expect-error: a plain ES module without types
import { collect } from "../../scripts/collect-data.mjs";

import { REPO } from "./paths";

// The SQL printed beside each chart drawn from a JSON file must return the rows the page computes.
// Both run here on the repository's own files: the SQL with DuckDB (the repository's Python
// environment has it), the rows with the page's function.
const python = [join(REPO, ".venv", "bin", "python"), "python3"].find((p) => {
  try {
    execFileSync(p, ["-c", "import duckdb"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
});

const all = Object.entries(queries).filter(([, q]) => typeof q === "object" && q !== null && "sql" in q) as [string, queries.JsonQuery<unknown>][];

describe.skipIf(!python)("the SQL beside each JSON chart", () => {
  const out = mkdtempSync(join(tmpdir(), "goodhart-jsonsql-"));
  const index = collect({ root: REPO, overlays: [], out });

  it.each(all)("%s returns the rows the page draws", (_name, q) => {
    const entry = index.sources[q.source];
    if (!entry || !existsSync(join(out, entry.url))) return;
    const json = JSON.parse(readFileSync(join(out, entry.url), "utf8"));
    const expected = q.rows(json);
    const script = "import duckdb, json, sys; r = duckdb.sql(sys.argv[1]); print(json.dumps([dict(zip(r.columns, row)) for row in r.fetchall()]))";
    const got = JSON.parse(execFileSync(python!, ["-I", "-c", script, q.sql], { cwd: out, encoding: "utf8" })) as Record<string, unknown>[];
    expect(got.length).toBe(expected.length);
    got.forEach((row, i) => {
      for (const [key, value] of Object.entries(expected[i]!)) {
        if (typeof value === "number") expect(row[key] as number, `${key} of row ${i}`).toBeCloseTo(value, 9);
        else expect(row[key], `${key} of row ${i}`).toEqual(value);
      }
    });
  });
});
