// The live check: the published site and the live grader, read from a client that is neither the
// build machine nor the API's host (the live workflow runs it on a GitHub runner), in a real browser.
//
//   node scripts/live-check.mjs [--site https://mekala27-45.github.io/goodhart/] [--api https://goodhart-grader.vercel.app]
//
// It times the API's first answer after idle (asleep) before anything else touches it, opens every
// route and checks the statement is on it and the console stays clean, grades one query on the desk
// and waits for the live verdict, then times the API again (awake). The record goes to
// results/live_check.json at the repository root.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const site = arg("--site", "https://mekala27-45.github.io/goodhart/").replace(/\/?$/, "/");
const api = arg("--api", "https://goodhart-grader.vercel.app").replace(/\/$/, "");
const routes = ["", "methods/", "hacks/", "verifier/", "data/", "desk/", "sizing/", "models/", "report/"];
// The first sentence of the statement is enough to find it; the whole of it is checked by the gates.
const statement = "goodhart is a portfolio demonstration.";

async function timedHealth() {
  const t0 = performance.now();
  try {
    const r = await fetch(`${api}/health`, { signal: AbortSignal.timeout(60_000) });
    return { status: r.status, ms: Math.round(performance.now() - t0), ok: r.ok };
  } catch (err) {
    return { status: null, ms: Math.round(performance.now() - t0), ok: false, error: String(err) };
  }
}

const asleep = await timedHealth();
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
const errors = [];
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text().slice(0, 300));
});
const pages = [];
for (const route of routes) {
  const before = errors.length;
  const response = await page.goto(`${site}${route}`, { waitUntil: "networkidle", timeout: 90_000 });
  const text = await page.locator("body").innerText();
  pages.push({
    route: `/${route}`,
    status: response?.status() ?? null,
    statement: text.includes(statement),
    console_errors: errors.length - before,
  });
}

const desk = { source: null, graded: false, grade_id: null, error: null };
try {
  await page.goto(`${site}desk/`, { waitUntil: "networkidle", timeout: 90_000 });
  const source = page.getByTestId("desk-source");
  await source.waitFor({ timeout: 60_000 });
  await page.waitForFunction(
    () => document.querySelector('[data-testid="desk-source"]')?.getAttribute("data-source") !== "probing",
    null,
    { timeout: 60_000 },
  );
  desk.source = await source.getAttribute("data-source");
  if (desk.source === "live") {
    const picker = page.getByTestId("question-picker");
    const first = await picker.locator("option").nth(1).getAttribute("value");
    if (first) await picker.selectOption(first);
    await page.getByTestId("sql-editor").fill("SELECT 1 AS value");
    await page.getByTestId("mark").click();
    const grade = page.getByTestId("grade");
    await grade.waitFor({ timeout: 60_000 });
    desk.graded = (await grade.getAttribute("data-source")) === "live";
    desk.grade_id = (await grade.innerText()).match(/[0-9a-f]{8}-[0-9a-f-]{27}/)?.[0] ?? null;
  }
} catch (err) {
  desk.error = String(err).slice(0, 500);
}
await browser.close();
const awake = await timedHealth();

const record = {
  checked_at: new Date().toISOString(),
  site,
  api,
  client: process.env.GITHUB_ACTIONS ? `a GitHub runner, run ${process.env.GITHUB_RUN_ID}` : "a local machine",
  api_asleep: asleep,
  api_awake: awake,
  pages,
  desk,
  ok: pages.every((p) => p.status === 200 && p.statement) && desk.source === "live" && desk.graded && awake.ok,
};
writeFileSync(join(web, "..", "results", "live_check.json"), `${JSON.stringify(record, null, 1)}\n`);
console.log(JSON.stringify(record, null, 1));
process.exit(record.ok ? 0 : 1);
