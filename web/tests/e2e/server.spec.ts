import { type ChildProcess, spawn } from "node:child_process";
import { join } from "node:path";

import { expect, test } from "@playwright/test";

import { BASE, FIXTURE, HERE, routeApi } from "./helpers";

interface Started {
  child: ChildProcess;
  port: number;
  log: { method: string; path: string; status: number }[];
}

/** A fresh test server on a free port, its request log read from its output. */
async function start(): Promise<Started> {
  const child = spawn(process.execPath, [join(HERE, "..", "..", "scripts", "test-server.mjs"), "--port", "0", "--log"], { stdio: ["ignore", "pipe", "inherit"] });
  const log: Started["log"] = [];
  const port = await new Promise<number>((resolve, reject) => {
    let buffer = "";
    child.stdout!.on("data", (chunk: Buffer) => {
      buffer += chunk.toString();
      let cut = buffer.indexOf("\n");
      while (cut >= 0) {
        const line = JSON.parse(buffer.slice(0, cut)) as Record<string, unknown>;
        buffer = buffer.slice(cut + 1);
        if (typeof line.port === "number") resolve(line.port);
        else log.push(line as Started["log"][number]);
        cut = buffer.indexOf("\n");
      }
    });
    child.on("exit", (code) => reject(new Error(`the test server exited with ${code}`)));
  });
  return { child, port, log };
}

test("the test server refuses HEAD and answers its very first request with a 503", async () => {
  const s = await start();
  try {
    const url = `http://127.0.0.1:${s.port}${BASE}/data/index.json`;
    const head = await fetch(url, { method: "HEAD" });
    expect(head.status).toBe(405);
    const first = await fetch(url);
    expect(first.status).toBe(503);
    expect(first.headers.get("retry-after")).toBe("1");
    const second = await fetch(url);
    expect(second.status).toBe(200);
    expect(((await second.json()) as { schema: number }).schema).toBe(1);
    expect((await fetch(url, { method: "HEAD" })).status).toBe(405);
  } finally {
    s.child.kill();
  }
});

test("the site's fetch helper retries the 503 and never sends HEAD", async ({ page }) => {
  // The page comes from the fixture server; its data requests are passed to a fresh server whose
  // very first answer is the 503, so the first data request the site makes meets it.
  const s = await start();
  try {
    await routeApi(page, "down");
    await page.route(`${FIXTURE}${BASE}/data/**`, async (route) => {
      const target = route.request().url().replace(FIXTURE, `http://127.0.0.1:${s.port}`);
      const response = await route.fetch({ url: target });
      await route.fulfill({ response });
    });
    await page.goto(`${FIXTURE}${BASE}/verifier/`);
    await expect(page.getByTestId("mutation-chart")).toHaveAttribute("data-state", "ready");
    await expect(page.getByTestId("mutation-chart").getByTestId("chart-message")).toContainText("near misses");
    expect(s.log[0]).toMatchObject({ method: "GET", status: 503 });
    const first = s.log[0]!.path;
    expect(s.log.slice(1).some((r) => r.path === first && r.status === 200)).toBe(true);
    expect(s.log.filter((r) => r.method === "HEAD")).toEqual([]);
  } finally {
    s.child.kill();
  }
});
