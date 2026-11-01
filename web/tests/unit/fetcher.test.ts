import { afterEach, describe, expect, it } from "vitest";

import { fetchRetry, getJson, HttpError, resetFetch, setFetch, waitBefore } from "@/lib/fetcher";

type Call = { url: string; method: string };

function host(answers: (number | "network")[]): { calls: Call[] } {
  const calls: Call[] = [];
  let i = 0;
  setFetch(async (input, init) => {
    calls.push({ url: String(input), method: String(init?.method ?? "GET") });
    const next = answers[Math.min(i, answers.length - 1)]!;
    i += 1;
    if (next === "network") throw new TypeError("failed to fetch");
    return new Response(JSON.stringify({ ok: next }), { status: next, headers: next === 503 ? { "Retry-After": "0" } : {} });
  });
  return { calls };
}

const quick = { baseWaitMs: 1, maxWaitMs: 5 };

afterEach(() => resetFetch());

describe("the site's fetch helper", () => {
  it("retries a 503, the way a waking host answers first, and returns the answer after it", async () => {
    const { calls } = host([503, 200]);
    const body = await getJson<{ ok: number }>("/goodhart/data/index.json", quick);
    expect(body).toEqual({ ok: 200 });
    expect(calls.map((c) => c.method)).toEqual(["GET", "GET"]);
  });

  it("retries a network failure and a 502 or 504 too, then gives up with the last answer", async () => {
    host(["network", 502, 504, 503]);
    const r = await fetchRetry("/x", {}, { ...quick, attempts: 4 });
    expect(r.status).toBe(503);
  });

  it("does not retry an answer that is final, such as a 404", async () => {
    const { calls } = host([404, 200]);
    await expect(getJson("/missing.json", quick)).rejects.toBeInstanceOf(HttpError);
    expect(calls).toHaveLength(1);
  });

  it("never sends HEAD: asking for one fails before any request leaves", async () => {
    const { calls } = host([200]);
    await expect(fetchRetry("/x", { method: "HEAD" })).rejects.toThrow("never sends HEAD");
    expect(calls).toHaveLength(0);
  });

  it("throws when no attempt produced an answer", async () => {
    host(["network"]);
    await expect(fetchRetry("/x", {}, { ...quick, attempts: 2 })).rejects.toThrow("failed to fetch");
  });

  it("waits as Retry-After says, capped, and otherwise doubles", () => {
    const o = { attempts: 4, timeoutMs: 1000, baseWaitMs: 300, maxWaitMs: 4000 };
    expect(waitBefore(2, "1", o)).toBe(1000);
    expect(waitBefore(2, "60", o)).toBe(4000);
    expect(waitBefore(2, null, o)).toBe(300);
    expect(waitBefore(3, null, o)).toBe(600);
    expect(waitBefore(9, "", o)).toBe(4000);
  });
});
