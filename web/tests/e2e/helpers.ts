import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Page, Route } from "@playwright/test";

export { STATEMENT } from "../../src/lib/statement";

export const BASE = "/goodhart";
const port = Number(process.env.GOODHART_E2E_PORT ?? 4173);
/** The server with the end to end fixtures laid over the repository's data. */
export const FIXTURE = `http://127.0.0.1:${port}`;
/** The server with the repository's data alone. */
export const REPO = `http://127.0.0.1:${port + 1}`;
export const API = "https://goodhart-grader.vercel.app";
export const ROUTES = ["/", "/methods/", "/hacks/", "/verifier/", "/data/", "/desk/", "/sizing/", "/models/", "/report/"];

export const HERE = dirname(fileURLToPath(import.meta.url));

export function apiFixture<T = unknown>(name: string): T {
  return JSON.parse(readFileSync(join(HERE, "..", "fixtures", "api", `${name}.json`), "utf8")) as T;
}

/**
 * Stand in for the grader API. "live" answers /health, /v1/questions and POST /v1/grade with fixtures
 * shaped like the real responses; "down" fails every request, the way an asleep host times out.
 * Returns the requests the page made, so a test can check what was sent.
 */
export async function routeApi(page: Page, mode: "live" | "down"): Promise<{ method: string; url: string; body: string | null }[]> {
  const seen: { method: string; url: string; body: string | null }[] = [];
  await page.route(`${API}/**`, async (route: Route) => {
    const req = route.request();
    seen.push({ method: req.method(), url: req.url(), body: req.postData() });
    if (mode === "down") return route.abort("connectionrefused");
    const path = new URL(req.url()).pathname;
    const json = (body: unknown) => route.fulfill({ status: 200, contentType: "application/json", headers: { "access-control-allow-origin": "*" }, body: JSON.stringify(body) });
    if (req.method() === "GET" && path === "/health") return json(apiFixture("health"));
    if (req.method() === "GET" && path === "/v1/questions") return json(apiFixture("questions"));
    if (req.method() === "POST" && path === "/v1/grade") return json(apiFixture("grade"));
    return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "no such route" }) });
  });
  return seen;
}

/** Page errors and failed console messages, gathered while a test runs. */
export function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}
