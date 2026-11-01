import { existsSync, mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import type { Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

// @ts-expect-error: a plain ES module without types
import { serve } from "../../scripts/test-server.mjs";

let server: Server;
let base: string;

beforeAll(async () => {
  const root = mkdtempSync(join(tmpdir(), "goodhart-serve-"));
  mkdirSync(join(root, "data"), { recursive: true });
  mkdirSync(join(root, "desk"), { recursive: true });
  writeFileSync(join(root, "index.html"), "<p>home</p>");
  writeFileSync(join(root, "desk", "index.html"), "<p>desk</p>");
  writeFileSync(join(root, "data", "index.json"), '{"from":"out"}');
  const overlay = mkdtempSync(join(tmpdir(), "goodhart-overlay-"));
  writeFileSync(join(overlay, "index.json"), '{"from":"overlay"}');
  const started = await serve({ port: 0, root, data: overlay, base: "/goodhart" });
  server = started.server;
  base = `http://127.0.0.1:${started.port}`;
  expect(existsSync(root)).toBe(true);
});

afterAll(() => server?.close());

describe("the test server", () => {
  it("refuses HEAD with a 405 before anything else, and answers the very first request with a 503", async () => {
    expect((await fetch(`${base}/goodhart/`, { method: "HEAD" })).status).toBe(405);
    const first = await fetch(`${base}/goodhart/`);
    expect(first.status).toBe(503);
    expect(first.headers.get("retry-after")).toBe("1");
    expect((await fetch(`${base}/goodhart/`)).status).toBe(200);
  });

  it("serves the export under the base path, with directory indexes and redirects", async () => {
    expect(await (await fetch(`${base}/goodhart/desk/`)).text()).toBe("<p>desk</p>");
    const bare = await fetch(`${base}/goodhart/desk`, { redirect: "manual" });
    expect(bare.status).toBe(301);
    expect(bare.headers.get("location")).toBe("/goodhart/desk/");
    expect((await fetch(`${base}/elsewhere/`)).status).toBe(404);
    expect((await fetch(`${base}/goodhart/`, { method: "POST" })).status).toBe(405);
  });

  it("serves the data overlay in place of out/data", async () => {
    expect(await (await fetch(`${base}/goodhart/data/index.json`)).json()).toEqual({ from: "overlay" });
  });
});
