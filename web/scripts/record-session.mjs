// Records the desk's fallback from the live grader API: GET /health, GET /v1/questions, and one
// POST /v1/grade for every try in scripts/tries.json, written to public/data/recorded_session.json
// in the same format scripts/record_local.py writes, with placeholder false. Each grade it records
// is a real grade in the API's grade log and counts against this client's hourly budget.
//
//   node scripts/record-session.mjs [--api https://goodhart-grader.vercel.app]
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const flag = process.argv.indexOf("--api");
const api = (flag > 0 ? process.argv[flag + 1] : "https://goodhart-grader.vercel.app").replace(/\/$/, "");
const tries = JSON.parse(readFileSync(join(web, "scripts", "tries.json"), "utf8"));

// The same habit as the site's helper: a waking host answers 503, so ask again; never send HEAD.
async function ask(path, init = {}, attempts = 4) {
  for (let i = 1; ; i += 1) {
    const r = await fetch(`${api}${path}`, init);
    if (![502, 503, 504].includes(r.status) || i >= attempts) return r;
    await new Promise((ok) => setTimeout(ok, 1500 * i));
  }
}

const health = await (await ask("/health")).json();
if (health.status !== "ok") throw new Error(`the API is not healthy: ${JSON.stringify(health)}`);
const questions = await (await ask("/v1/questions")).json();
const grades = [];
for (const t of tries) {
  const r = await ask("/v1/grade", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question_id: t.question_id, sql: t.sql }) }, 2);
  grades.push({ request: { question_id: t.question_id, sql: t.sql }, status: r.status, response: await r.json() });
  console.log(`${t.question_id} ${t.label}: ${r.status}`);
}
const session = {
  recorded: true,
  placeholder: false,
  recorded_at: new Date().toISOString().slice(0, 10),
  recorded_from: `the live API at ${api}`,
  api,
  note: "Recorded by web/scripts/record-session.mjs.",
  health,
  questions,
  tries,
  grades,
};
writeFileSync(join(web, "public", "data", "recorded_session.json"), `${JSON.stringify(session, null, 1)}\n`);
console.log(`recorded ${grades.length} grades from ${api}`);
