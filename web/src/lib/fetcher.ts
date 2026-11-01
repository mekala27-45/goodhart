// The one way this site asks a server for anything: the published data files, the parquet grades
// and the grader API all come through here.
//
// Two habits of the hosts it runs against are built in. A host that is waking answers 503 (Vercel's
// first answer after a cold start, and the test server's very first answer), so a 502, 503 or 504
// is retried with a short backoff that honours Retry-After. And some hosts refuse HEAD, so this
// helper never sends one: a caller that asks for HEAD gets an error before anything leaves.

export class HttpError extends Error {
  readonly status: number;
  readonly url: string;
  constructor(status: number, url: string, detail?: string) {
    super(detail ? `${url} answered ${status}: ${detail}` : `${url} answered ${status}`);
    this.name = "HttpError";
    this.status = status;
    this.url = url;
  }
}

export interface RetryOptions {
  /** How many times to ask in all, the first included. */
  attempts?: number;
  /** A limit on each attempt, in milliseconds. */
  timeoutMs?: number;
  /** The wait before the second attempt; each later wait doubles, up to maxWaitMs. */
  baseWaitMs?: number;
  maxWaitMs?: number;
  /** A caller's own signal: aborting it stops the retries. */
  signal?: AbortSignal;
}

export const RETRY_STATUS = new Set([502, 503, 504]);

const DEFAULTS: Required<Omit<RetryOptions, "signal">> = {
  attempts: 4,
  timeoutMs: 20_000,
  baseWaitMs: 350,
  maxWaitMs: 4_000,
};

/** The fetch this module calls; tests replace it. */
let fetchImpl: typeof fetch = (input, init) => fetch(input, init);
export function setFetch(fn: typeof fetch): void {
  fetchImpl = fn;
}
export function resetFetch(): void {
  fetchImpl = (input, init) => fetch(input, init);
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** How long to wait before attempt number `next` (2 or more). Retry-After in seconds wins, capped. */
export function waitBefore(next: number, retryAfter: string | null, o: Required<Omit<RetryOptions, "signal">>): number {
  const seconds = retryAfter !== null && retryAfter.trim() !== "" ? Number(retryAfter) : Number.NaN;
  if (Number.isFinite(seconds) && seconds >= 0) return Math.min(seconds * 1000, o.maxWaitMs);
  return Math.min(o.baseWaitMs * 2 ** (next - 2), o.maxWaitMs);
}

/**
 * fetch, retried on a waking host. Returns the last response whatever its status (a 404 is an
 * answer, not a failure); throws only when no attempt produced a response.
 */
export async function fetchRetry(url: string, init: RequestInit = {}, options: RetryOptions = {}): Promise<Response> {
  const method = (init.method ?? "GET").toUpperCase();
  if (method === "HEAD") throw new Error("this site never sends HEAD: a host it runs on refuses it");
  const o = { ...DEFAULTS, ...options };
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= o.attempts; attempt += 1) {
    if (options.signal?.aborted) throw options.signal.reason ?? new DOMException("aborted", "AbortError");
    const controller = new AbortController();
    const onAbort = () => controller.abort(options.signal?.reason);
    options.signal?.addEventListener("abort", onAbort, { once: true });
    const timer = setTimeout(() => controller.abort(new DOMException("timed out", "TimeoutError")), o.timeoutMs);
    try {
      const response = await fetchImpl(url, { ...init, method, signal: controller.signal });
      if (RETRY_STATUS.has(response.status) && attempt < o.attempts) {
        await response.body?.cancel().catch(() => undefined);
        await sleep(waitBefore(attempt + 1, response.headers.get("Retry-After"), o));
        continue;
      }
      return response;
    } catch (err) {
      lastError = err;
      if (options.signal?.aborted) throw err;
      if (attempt < o.attempts) {
        await sleep(waitBefore(attempt + 1, null, o));
        continue;
      }
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(`${url} did not answer`);
}

async function ok(url: string, init?: RequestInit, options?: RetryOptions): Promise<Response> {
  const r = await fetchRetry(url, init, options);
  if (!r.ok) throw new HttpError(r.status, url);
  return r;
}

export async function getJson<T>(url: string, options?: RetryOptions): Promise<T> {
  return (await (await ok(url, undefined, options)).json()) as T;
}

export async function getText(url: string, options?: RetryOptions): Promise<string> {
  return (await ok(url, undefined, options)).text();
}

export async function getBytes(url: string, options?: RetryOptions): Promise<ArrayBuffer> {
  return (await ok(url, undefined, options)).arrayBuffer();
}
