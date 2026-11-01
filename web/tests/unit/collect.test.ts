import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { beforeAll, describe, expect, it } from "vitest";

// @ts-expect-error: a plain ES module without types
import { collect, pickScripts, publicQuestion } from "../../scripts/collect-data.mjs";

function put(root: string, rel: string, content: string) {
  const path = join(root, rel);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content);
}

const gold = { sql: "select 1", rows: [[1]], columns: [], label: false, ordered: false, tables: [] };
const question = (id: string, extra = {}) => JSON.stringify({ id, question: `q ${id}`, family: "lookup", warehouse: "doorstep", template: "t", measure: "m", gold, refusal_reason: null, split: "readback_test", ...extra });

let root: string;
let overlay: string;
let out: string;
let index: Record<string, any>;

beforeAll(() => {
  const base = mkdtempSync(join(tmpdir(), "goodhart-collect-"));
  root = join(base, "repo");
  overlay = join(base, "fixtures");
  out = join(base, "out");
  put(root, "results/mutation.json", JSON.stringify({ kept_twins: [] }));
  put(root, "evaluation/plan.yaml", "plan: test\ncompute:\n  quota_plan:\n    probe: 0.3\n");
  put(root, "data/questions/readback_test.jsonl", [question("rb-1"), question("rb-2")].join("\n") + "\n");
  put(root, "data/questions/dev.jsonl", question("d-1", { split: "dev" }) + "\n");
  put(root, "docs/models/m4.md", "# M4\n");
  // A run in the repository, and one in the overlay with completions, grades and a request.
  put(root, "results/runs/r-honest/ledger.jsonl", JSON.stringify({ request: "r-honest", stage: "grpo", arm: "honest", status: "done", steps: 50 }) + "\n");
  put(overlay, "requests/r-lenient.json", JSON.stringify({ id: "r-lenient", stage: "grpo", model: "small", arm: "lenient", reward: "lenient", hyperparameters: {} }));
  put(overlay, "results/runs/r-lenient/monitor.jsonl", JSON.stringify({ step: 1, reward: 0.5 }) + "\n");
  const completion = (checkpoint: string, sample: number, temperature: number, text: string) =>
    JSON.stringify({ item_id: "d-1", request: "r-lenient", checkpoint, sample, temperature, text, tokens: 3 });
  put(overlay, "results/runs/r-lenient/completions/dev__step-25.jsonl", completion("step-25", 0, 0, "old") + "\n");
  put(overlay, "results/runs/r-lenient/completions/dev__step-50.jsonl", [completion("step-50", 1, 0.8, "second"), completion("step-50", 0, 0, "greedy")].join("\n") + "\n");
  put(overlay, "results/grades/index.json", JSON.stringify({ "r-lenient/dev__step-50": { source: "x", grades: "y", rows: 2, set: "dev" } }));
  put(overlay, "results/grades/r-lenient/dev__step-50.parquet", "PAR1");
  // A recorded session already in the output folder survives the collection.
  put(out, "recorded_session.json", "{}");
  put(out, "stale.json", "{}");
  index = collect({ root, overlays: [overlay], out });
});

describe("collect-data", () => {
  it("copies what exists and lists what does not", () => {
    expect(index.sources["results/mutation.json"]).toMatchObject({ url: "results/mutation.json", origin: "repository" });
    expect(index.missing).toContain("results/sizing.json");
    expect(index.missing).toContain("report/briefing.md");
    expect(JSON.parse(readFileSync(join(out, "index.json"), "utf8")).schema).toBe(1);
  });

  it("publishes YAML as JSON with the same content", () => {
    expect(index.sources["evaluation/plan.yaml"].url).toBe("evaluation/plan.json");
    expect(JSON.parse(readFileSync(join(out, "evaluation/plan.json"), "utf8"))).toEqual({ plan: "test", compute: { quota_plan: { probe: 0.3 } } });
  });

  it("never publishes the gold of a held-out question", () => {
    const text = readFileSync(join(out, "data/questions/readback_test.json"), "utf8");
    expect(text).not.toContain("gold");
    expect(text).not.toContain("select 1");
    expect(text).not.toContain("refusal_reason");
    expect(JSON.parse(text)[0]).toEqual({ set: "readback_test", id: "rb-1", question: "q rb-1", family: "lookup", warehouse: "doorstep", template: "t", measure: "m" });
    expect(publicQuestion({ id: "x", gold: {}, question: "q" }, "test")).toEqual({ set: "test", id: "x", question: "q" });
  });

  it("keeps the recorded session and clears everything else from the last build", () => {
    expect(existsSync(join(out, "recorded_session.json"))).toBe(true);
    expect(existsSync(join(out, "stale.json"))).toBe(false);
  });

  it("lists runs with their files, request fields and the fixture origin", () => {
    expect(index.runs["r-honest"]).toMatchObject({ stage: "grpo", arm: "honest", status: "done", steps: 50 });
    expect(index.runs["r-lenient"]).toMatchObject({ stage: "grpo", arm: "lenient", reward: "lenient" });
    expect(index.runs["r-lenient"].files).toMatchObject({ monitor: "results/runs/r-lenient/monitor.jsonl", request: "requests/r-lenient.json" });
    expect(index.sources["results/runs/r-lenient/monitor.jsonl"].origin).toBe("fixture");
  });

  it("writes one script per run, checkpoint and item, the greedy answer first, the last dev checkpoint only", () => {
    const dev = JSON.parse(readFileSync(join(out, "scripts/dev.json"), "utf8"));
    expect(dev.items["d-1"].question).toBe("q d-1");
    expect(dev.items["d-1"].answers).toHaveLength(1);
    expect(dev.items["d-1"].answers[0]).toMatchObject({ checkpoint: "step-50", text: "greedy", origin: "fixture" });
    expect(index.scripts.dev).toMatchObject({ items: 1, runs: ["r-lenient"] });
    expect(pickScripts([{ request: "a", checkpoint: "c", item_id: "i", sample: 3, temperature: 0.8 }, { request: "a", checkpoint: "c", item_id: "i", sample: 1, temperature: 0.8 }])[0].sample).toBe(1);
  });

  it("copies every graded file the merged index lists", () => {
    expect(index.grades["r-lenient/dev__step-50"]).toMatchObject({ url: "results/grades/r-lenient/dev__step-50.parquet", run: "r-lenient", set: "dev", rows: 2, origin: "fixture" });
    expect(readFileSync(join(out, "results/grades/r-lenient/dev__step-50.parquet"), "utf8")).toBe("PAR1");
  });

  it("lists the model cards it rendered", () => {
    expect(index.docs.models).toEqual([{ name: "m4", url: "docs/models/m4.md" }]);
    expect(index.docs.validation).toEqual([]);
  });
});
