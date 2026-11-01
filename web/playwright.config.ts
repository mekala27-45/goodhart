import { existsSync } from "node:fs";

import { defineConfig, devices } from "@playwright/test";

// The exported site under its base path, served by scripts/test-server.mjs, which refuses HEAD and
// answers its very first request with a 503. Two servers: one with the end to end fixtures laid over
// the repository's data (the runs that do not exist yet), one with the repository's data alone (the
// "not yet run" states). Build first with `npm run build`; `npm run e2e` collects the fixtures.
//
// This sandbox has Chromium outside Playwright's cache; set GOODHART_CHROMIUM elsewhere, or leave it
// unset where `npx playwright install chromium` has run.
const chromium = process.env.GOODHART_CHROMIUM ?? "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome";
export const FIXTURE_PORT = Number(process.env.GOODHART_E2E_PORT ?? 4173);
export const REPO_PORT = FIXTURE_PORT + 1;

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 2,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/results.json" }]],
  use: {
    baseURL: `http://127.0.0.1:${FIXTURE_PORT}`,
    trace: "retain-on-failure",
    ...(existsSync(chromium) ? { launchOptions: { executablePath: chromium } } : {}),
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1360, height: 900 } }, testIgnore: /phone\.spec\.ts/ },
    { name: "phone", use: { ...devices["Pixel 7"], viewport: { width: 390, height: 844 } }, testMatch: /phone\.spec\.ts/ },
  ],
  webServer: [
    {
      command: `node scripts/test-server.mjs --port ${FIXTURE_PORT} --data tests/.fixture-data`,
      url: `http://127.0.0.1:${FIXTURE_PORT}/goodhart/`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: `node scripts/test-server.mjs --port ${REPO_PORT}`,
      url: `http://127.0.0.1:${REPO_PORT}/goodhart/`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
  ],
});
