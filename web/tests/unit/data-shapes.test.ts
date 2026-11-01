import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { TRIES, type RecordedSession } from "@/lib/api";
import { decodeParquet } from "@/lib/parquet";
import { STATEMENT } from "@/lib/statement";

import { WEB } from "./paths";

const read = <T>(rel: string): T => JSON.parse(readFileSync(join(WEB, rel), "utf8")) as T;

// The keys POST /v1/grade answers with (packages/api/src/goodhart_api/app.py).
const GRADE_KEYS = ["grade_id", "question_id", "set", "kind", "executed", "error", "columns", "rows", "row_count", "truncated", "pays", "checks", "flags", "verifier", "statement"];
const DETECTORS = ["literal_only", "copied_value", "unfiltered", "refusal_answerable", "valid_but_wrong"];

describe("the recorded session", () => {
  const s = read<RecordedSession>("public/data/recorded_session.json");

  it("says it is recorded, and where from", () => {
    expect(s.recorded).toBe(true);
    expect(typeof s.placeholder).toBe("boolean");
    expect(s.recorded_from.length).toBeGreaterThan(10);
  });

  it("holds responses in the exact shape the API answers with", () => {
    expect(s.health).toMatchObject({ status: "ok", statement: STATEMENT });
    expect(s.questions.statement).toBe(STATEMENT);
    for (const q of s.questions.questions) expect(Object.keys(q).sort()).toEqual(["family", "id", "question", "set", "warehouse"]);
    for (const g of s.grades) {
      expect(Object.keys(g.response).sort()).toEqual([...GRADE_KEYS].sort());
      expect(Object.keys(g.response.pays).sort()).toEqual(["honest", "lazy", "lenient", "refusal"]);
      expect(Object.keys(g.response.checks).sort()).toEqual(["confirmed", "original", "twins_passed", "twins_total"]);
      expect(Object.keys(g.response.flags).sort()).toEqual([...DETECTORS].sort());
      expect(g.response.statement).toBe(STATEMENT);
      expect(g.response.question_id).toBe(g.request.question_id);
    }
  });

  it("has the five tries for each question it covers, each with its recorded grade", () => {
    const byQuestion = new Map<string, string[]>();
    for (const t of s.tries) byQuestion.set(t.question_id, [...(byQuestion.get(t.question_id) ?? []), t.label]);
    expect(byQuestion.size).toBeGreaterThan(0);
    for (const labels of byQuestion.values()) expect([...labels].sort()).toEqual([...TRIES].sort());
    for (const t of s.tries) expect(s.grades.some((g) => g.request.question_id === t.question_id && g.request.sql === t.sql)).toBe(true);
  });

  it("never carries the gold of a question", () => {
    expect(JSON.stringify(s)).not.toContain('"gold"');
  });
});

describe("the API fixtures for the end to end tests", () => {
  it("have the shape of the real responses", () => {
    const grade = read<Record<string, unknown>>("tests/fixtures/api/grade.json");
    expect(Object.keys(grade).sort()).toEqual([...GRADE_KEYS].sort());
    expect(read<{ statement: string }>("tests/fixtures/api/health.json").statement).toBe(STATEMENT);
    expect(read<{ statement: string }>("tests/fixtures/api/questions.json").statement).toBe(STATEMENT);
  });
});

describe("a grades parquet file, read the way the browser reads it", () => {
  it("decodes every column with its type and every row", async () => {
    const index = read<Record<string, { rows: number }>>("tests/fixtures/repo/results/grades/index.json");
    const [key, entry] = Object.entries(index)[0]!;
    const buf = readFileSync(join(WEB, "tests/fixtures/repo/results/grades", `${key}.parquet`));
    const decoded = await decodeParquet(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(decoded.rows).toHaveLength(entry.rows);
    const types = Object.fromEntries(decoded.columns.map((c) => [c.name, c.type]));
    expect(types).toMatchObject({ item_id: "string", request: "string", checkpoint: "string", set: "string", sample: "int", temperature: "double", refuse_item: "bool", confirmed: "bool", tokens: "int", flag_copied_value: "bool" });
    expect(typeof decoded.rows[0]!.sample).toBe("number");
  });
});
