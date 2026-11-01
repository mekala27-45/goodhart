// collect-data: copies what the repository holds into public/data, and writes public/data/index.json
// listing what was found and what was not. `npm run build` runs it before `next build`.
//
// The pages read nothing else. A section whose file is missing says "not yet run" in words, so the
// site builds and renders with any subset present: on the day it was written most GPU results did
// not exist yet. Three rules hold for everything copied:
//
//   the gold answer of a held-out question never leaves the repository (the field is dropped here,
//   before anything is published);
//   YAML is published as JSON with the same content, so the pages need no YAML parser;
//   a model's completions are published only as one script per question, run and checkpoint (the
//   greedy answer, or the first sample), for the desk and the hacks page to print with its marks.
//
//   node scripts/collect-data.mjs [--root ..] [--overlay dir]... [--out public/data]
//
// An overlay is a folder laid out like the repository whose files win over the repository's; the
// end to end tests use one (tests/fixtures/repo) for the runs that do not exist yet, and every file
// that came from an overlay is marked origin "fixture" in the index.
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import YAML from "yaml";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** Files the out folder keeps between builds: authored here, not copied. */
export const KEEP = new Set(["recorded_session.json"]);

/** The fixed sources, by repository path, and how each is published. */
export const SOURCES = [
  ["results/mutation.json", "json"],
  ["results/contamination.json", "json"],
  ["results/profile.json", "json"],
  ["results/sizing.json", "json"],
  ["results/gpu_probe.json", "json"],
  ["results/selection.json", "json"],
  ["results/audit.json", "json"],
  ["results/parity.json", "json"],
  ["results/latency.json", "json"],
  ["results/manifest.json", "json"],
  ["results/judge.json", "json"],
  ["results/live_check.json", "json"],
  ["data/questions/manifest.json", "json"],
  ["data/twins.lock.json", "json"],
  ["evaluation/plan.yaml", "yaml"],
  ["evaluation/predictions.yaml", "yaml"],
  ["evaluation/hypotheses.yaml", "yaml"],
  ["config/monitor.yaml", "yaml"],
  ["report/briefing.md", "text"],
  ["data/questions/readback_test.jsonl", "questions"],
];

/** Markdown folders rendered on the pages: the cards and reports on /models, the incidents on /hacks. */
export const DOC_FOLDERS = { "docs/models": "models", "docs/validation": "validation", "docs/incidents": "incidents" };

/** What a run folder may hold that the pages read. */
export const RUN_FILES = {
  ledger: "ledger.jsonl",
  monitor: "monitor.jsonl",
  alerts: "alerts.json",
  probe: "probe.json",
  provenance: "provenance.json",
  trainer: "metrics/trainer.jsonl",
};

/** The sets whose completions become scripts. Dev keeps only each run's last checkpoint. */
export const SCRIPT_SETS = ["readback_test", "test", "template_test", "dev"];

/** The fields of a question that may be published. The gold, the refusal reason and the SQL may not. */
export const QUESTION_FIELDS = ["id", "question", "family", "warehouse", "template", "measure", "unit_asked", "source"];

const MAX_SCRIPT_CHARS = 4000;

function parseArgs(argv) {
  const out = { root: resolve(WEB, ".."), overlays: [], out: join(WEB, "public", "data") };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === "--root" && value) out.root = resolve(value);
    else if (flag === "--overlay" && value) out.overlays.push(resolve(value));
    else if (flag === "--out" && value) out.out = resolve(value);
    else throw new Error(`unknown argument ${flag}`);
    i += 1;
  }
  return out;
}

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

function readLines(text) {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

/** Where a repository path is found: the first overlay that has it, then the root. */
export function locate(rel, ctx) {
  for (const overlay of ctx.overlays) {
    const abs = join(overlay, rel);
    if (existsSync(abs)) return { abs, origin: "fixture" };
  }
  const abs = join(ctx.root, rel);
  return existsSync(abs) ? { abs, origin: "repository" } : null;
}

function listDir(rel, ctx) {
  const names = new Map();
  for (const base of [ctx.root, ...[...ctx.overlays].reverse()]) {
    const dir = join(base, rel);
    if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
    for (const name of readdirSync(dir)) names.set(name, true);
  }
  return [...names.keys()].sort();
}

function clean(out) {
  mkdirSync(out, { recursive: true });
  for (const name of readdirSync(out)) {
    if (KEEP.has(name)) continue;
    rmSync(join(out, name), { recursive: true, force: true });
  }
}

function write(out, url, content) {
  const target = join(out, url);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, content);
  return target;
}

/** A held-out question as it may be published: no gold, no refusal reason, no SQL. */
export function publicQuestion(row, set) {
  const q = { set };
  for (const field of QUESTION_FIELDS) if (row[field] !== undefined) q[field] = row[field];
  return q;
}

function questionsOf(set, ctx) {
  if (ctx.questionCache.has(set)) return ctx.questionCache.get(set);
  const found = locate(`data/questions/${set}.jsonl`, ctx);
  const map = new Map();
  if (found) for (const row of readLines(readFileSync(found.abs, "utf8"))) map.set(row.id, publicQuestion(row, set));
  ctx.questionCache.set(set, map);
  return map;
}

function stepOf(checkpoint) {
  const m = /step-(\d+)/.exec(String(checkpoint));
  return m ? Number(m[1]) : null;
}

/** One script per run, checkpoint and item: the greedy answer, else the lowest sample. */
export function pickScripts(rows) {
  const best = new Map();
  for (const r of rows) {
    const key = `${r.request}|${r.checkpoint}|${r.item_id}`;
    const have = best.get(key);
    const rank = (x) => (Number(x.temperature) === 0 ? -1 : Number(x.sample));
    if (!have || rank(r) < rank(have)) best.set(key, r);
  }
  return [...best.values()];
}

function collectRun(id, ctx, index, scripts) {
  const runRel = `results/runs/${id}`;
  const entry = { files: {}, completions: [] };
  for (const [name, rel] of Object.entries(RUN_FILES)) {
    const found = locate(`${runRel}/${rel}`, ctx);
    if (!found) continue;
    const url = `${runRel}/${rel}`;
    const buf = readFileSync(found.abs);
    write(ctx.out, url, buf);
    entry.files[name] = url;
    index.sources[url] = { url, bytes: buf.length, sha256: sha256(buf), origin: found.origin };
  }
  const request = locate(`requests/${id}.json`, ctx);
  if (request) {
    const buf = readFileSync(request.abs);
    const url = `requests/${id}.json`;
    write(ctx.out, url, buf);
    entry.files.request = url;
    index.sources[url] = { url, bytes: buf.length, sha256: sha256(buf), origin: request.origin };
    try {
      const req = JSON.parse(buf.toString("utf8"));
      Object.assign(entry, { stage: req.stage ?? null, model: req.model ?? null, arm: req.arm ?? null, reward: req.reward ?? null, seed: req.seed ?? null });
      if (req.hyperparameters?.method) entry.method = req.hyperparameters.method;
    } catch {
      // A request that does not parse is still listed; the ledger below fills what it can.
    }
  }
  const ledger = locate(`${runRel}/ledger.jsonl`, ctx);
  if (ledger) {
    const rows = readLines(readFileSync(ledger.abs, "utf8"));
    const last = rows.at(-1);
    if (last) {
      entry.stage ??= last.stage ?? null;
      entry.model ??= last.model ?? null;
      entry.arm = entry.arm ?? last.arm ?? null;
      entry.status = last.status ?? null;
      entry.steps = last.steps ?? null;
    }
  }
  // Completions become scripts. A run's dev export keeps its last checkpoint only.
  const folder = locate(`${runRel}/completions`, ctx);
  if (folder && statSync(folder.abs).isDirectory()) {
    const files = readdirSync(folder.abs).filter((f) => f.endsWith(".jsonl") && !f.startsWith("paraphrase__")).sort();
    const devFiles = files.filter((f) => f.startsWith("dev__"));
    const lastDev = devFiles.sort((a, b) => (stepOf(a) ?? -1) - (stepOf(b) ?? -1)).at(-1);
    for (const file of files) {
      const set = file.split("__")[0];
      entry.completions.push(file.replace(/\.jsonl$/, ""));
      if (!SCRIPT_SETS.includes(set)) continue;
      if (set === "dev" && file !== lastDev) continue;
      const rows = pickScripts(readLines(readFileSync(join(folder.abs, file), "utf8")));
      const questions = questionsOf(set, ctx);
      const bucket = scripts.get(set) ?? { set, items: {} };
      for (const r of rows) {
        const q = questions.get(r.item_id);
        const item = (bucket.items[r.item_id] ??= {
          question: q?.question ?? null,
          family: q?.family ?? null,
          warehouse: q?.warehouse ?? null,
          answers: [],
        });
        item.answers.push({
          run: id,
          request: r.request,
          checkpoint: r.checkpoint,
          sample: r.sample,
          temperature: r.temperature,
          tokens: r.tokens ?? 0,
          text: String(r.text ?? "").slice(0, MAX_SCRIPT_CHARS),
          origin: folder.origin,
        });
      }
      scripts.set(set, bucket);
    }
  }
  index.runs[id] = entry;
}

function collectGrades(ctx, index) {
  const merged = {};
  for (const base of [ctx.root, ...[...ctx.overlays].reverse()]) {
    const path = join(base, "results", "grades", "index.json");
    if (!existsSync(path)) continue;
    const listed = JSON.parse(readFileSync(path, "utf8"));
    for (const [key, value] of Object.entries(listed)) merged[key] = { ...value, origin: base === ctx.root ? "repository" : "fixture" };
  }
  const keys = Object.keys(merged).sort();
  if (!keys.length) return;
  const indexUrl = "results/grades/index.json";
  const indexBuf = Buffer.from(JSON.stringify(merged, null, 1) + "\n");
  write(ctx.out, indexUrl, indexBuf);
  index.sources[indexUrl] = { url: indexUrl, bytes: indexBuf.length, sha256: sha256(indexBuf), origin: keys.some((k) => merged[k].origin === "fixture") ? "fixture" : "repository" };
  for (const key of keys) {
    const rel = `results/grades/${key}.parquet`;
    const found = locate(rel, ctx);
    if (!found) {
      index.missing.push(rel);
      continue;
    }
    const buf = readFileSync(found.abs);
    write(ctx.out, rel, buf);
    const [run, name] = key.split("/");
    index.grades[key] = { url: rel, run, name, set: merged[key].set ?? name.split("__")[0], rows: merged[key].rows ?? null, bytes: buf.length, origin: found.origin };
  }
}

export function collect({ root, overlays = [], out }) {
  const ctx = { root, overlays, out, questionCache: new Map() };
  clean(out);
  const index = { schema: 1, collected_at: new Date().toISOString(), sources: {}, missing: [], runs: {}, grades: {}, scripts: {}, docs: { models: [], validation: [], incidents: [] } };

  for (const [rel, kind] of SOURCES) {
    const found = locate(rel, ctx);
    if (!found) {
      index.missing.push(rel);
      continue;
    }
    let url = rel;
    let buf = readFileSync(found.abs);
    if (kind === "yaml") {
      url = rel.replace(/\.ya?ml$/, ".json");
      buf = Buffer.from(JSON.stringify(YAML.parse(buf.toString("utf8")), null, 1) + "\n");
    } else if (kind === "questions") {
      url = rel.replace(/\.jsonl$/, ".json");
      const set = basename(rel, ".jsonl");
      const rows = readLines(buf.toString("utf8")).map((r) => publicQuestion(r, set));
      buf = Buffer.from(JSON.stringify(rows, null, 1) + "\n");
    } else if (kind === "json") {
      JSON.parse(buf.toString("utf8"));
    }
    write(out, url, buf);
    index.sources[rel] = { url, bytes: buf.length, sha256: sha256(buf), origin: found.origin };
  }

  for (const [folder, kind] of Object.entries(DOC_FOLDERS)) {
    for (const name of listDir(folder, ctx).filter((n) => n.endsWith(".md"))) {
      const rel = `${folder}/${name}`;
      const found = locate(rel, ctx);
      if (!found) continue;
      const buf = readFileSync(found.abs);
      write(out, rel, buf);
      index.sources[rel] = { url: rel, bytes: buf.length, sha256: sha256(buf), origin: found.origin };
      index.docs[kind].push({ name: name.replace(/\.md$/, ""), url: rel });
    }
  }

  const scripts = new Map();
  for (const id of listDir("results/runs", ctx)) {
    const found = locate(`results/runs/${id}`, ctx);
    if (found && statSync(found.abs).isDirectory()) collectRun(id, ctx, index, scripts);
  }
  for (const [set, bucket] of [...scripts.entries()].sort()) {
    const url = `scripts/${set}.json`;
    const buf = Buffer.from(JSON.stringify(bucket) + "\n");
    write(out, url, buf);
    const runs = [...new Set(Object.values(bucket.items).flatMap((i) => i.answers.map((a) => a.run)))].sort();
    index.scripts[set] = { url, items: Object.keys(bucket.items).length, runs, bytes: buf.length };
  }

  collectGrades(ctx, index);
  index.missing.sort();
  writeFileSync(join(out, "index.json"), JSON.stringify(index, null, 1) + "\n");
  return index;
}

const invoked = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invoked) {
  const args = parseArgs(process.argv.slice(2));
  const index = collect(args);
  const found = Object.keys(index.sources).length;
  const where = relative(process.cwd(), args.out) || ".";
  console.log(
    `collect-data: ${found} files, ${Object.keys(index.runs).length} runs, ${Object.keys(index.grades).length} graded files, ` +
      `${Object.keys(index.scripts).length} script sets into ${where}; not yet in the repository: ${index.missing.length}`,
  );
}
