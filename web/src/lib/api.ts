"use client";

// The grader API (packages/api) and its recorded stand in.
//
// The desk probes GET /health twice, a few seconds apart, each with a time limit: a Vercel container
// that has been asleep can take one answer to wake. If neither probe answers, the desk plays the
// recorded session in public/data/recorded_session.json for the rest of the visit and says so in
// words. A live grade that fails is shown as a failure, never quietly swapped for a recording.
import { fetchRetry, HttpError } from "./fetcher";
import { API, dataUrl } from "./site";

export interface Health {
  status: string;
  version?: string;
  stage?: string;
  database?: string;
  warehouses_lock?: string | null;
  twins_lock?: string;
  kept_twins?: string[];
  questions?: number;
  verifier?: string;
  statement: string;
}

export interface ApiQuestion {
  id: string;
  set: string;
  question: string;
  family: string;
  warehouse: string;
}

export interface QuestionList {
  questions: ApiQuestion[];
  statement: string;
}

/** POST /v1/grade, as packages/api/src/goodhart_api/app.py answers it. */
export interface Grade {
  grade_id: string | null;
  question_id: string;
  set: string;
  kind: string;
  executed: boolean;
  error: string | null;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  pays: Record<string, number>;
  checks: { original: boolean; twins_passed: number; twins_total: number; confirmed: boolean };
  flags: Record<string, boolean>;
  verifier: string;
  statement: string;
}

export const TRIES = ["print the number", "drop the filter", "return the whole table", "refuse", "write it properly"] as const;
export type TryLabel = (typeof TRIES)[number];

export interface Try {
  question_id: string;
  label: TryLabel;
  sql: string;
}

export interface RecordedGrade {
  request: { question_id: string; sql: string };
  status: number;
  response: Grade;
}

/** public/data/recorded_session.json. */
export interface RecordedSession {
  recorded: true;
  placeholder: boolean;
  recorded_at: string | null;
  recorded_from: string;
  api: string;
  note: string;
  health: Health;
  questions: QuestionList;
  tries: Try[];
  grades: RecordedGrade[];
}

export type Source = "live" | "recorded";

export const PROBE_TIMEOUT_MS = 6000;
export const PROBE_PAUSE_MS = 3000;

let probe: Promise<{ source: Source; health: Health | null }> | null = null;

async function probeOnce(): Promise<Health | null> {
  try {
    const r = await fetchRetry(`${API}/health`, { cache: "no-store" }, { attempts: 1, timeoutMs: PROBE_TIMEOUT_MS });
    if (!r.ok) return null;
    const body = (await r.json()) as Health;
    return body && body.status === "ok" ? body : null;
  } catch {
    return null;
  }
}

/** Live or recorded for this visit: two probes of /health, then the answer is kept. */
export function apiSource(): Promise<{ source: Source; health: Health | null }> {
  probe ??= (async () => {
    for (let attempt = 1; attempt <= 2; attempt += 1) {
      if (attempt > 1) await new Promise((r) => setTimeout(r, PROBE_PAUSE_MS));
      const health = await probeOnce();
      if (health) return { source: "live" as Source, health };
    }
    return { source: "recorded" as Source, health: null };
  })();
  return probe;
}

let recording: Promise<RecordedSession> | null = null;
export function recordedSession(): Promise<RecordedSession> {
  recording ??= fetchRetry(dataUrl("recorded_session.json")).then(async (r) => {
    if (!r.ok) throw new HttpError(r.status, "recorded_session.json");
    return (await r.json()) as RecordedSession;
  });
  return recording;
}

export async function liveQuestions(): Promise<QuestionList> {
  const r = await fetchRetry(`${API}/v1/questions`, {}, { attempts: 3, timeoutMs: 15_000 });
  if (!r.ok) throw new HttpError(r.status, `${API}/v1/questions`);
  return (await r.json()) as QuestionList;
}

/** One grade from the live API. A refusal by the API (a 4xx) comes back as its detail, not a grade. */
export async function liveGrade(questionId: string, sql: string): Promise<{ grade: Grade | null; error: string | null }> {
  const r = await fetchRetry(
    `${API}/v1/grade`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question_id: questionId, sql }) },
    { attempts: 2, timeoutMs: 30_000 },
  );
  if (!r.ok) {
    const body = (await r.json().catch(() => ({}))) as { detail?: unknown };
    const detail = typeof body.detail === "string" ? body.detail : `the API answered ${r.status}`;
    return { grade: null, error: detail };
  }
  return { grade: (await r.json()) as Grade, error: null };
}

const squash = (s: string) => s.replace(/\s+/g, " ").trim().toLowerCase();

/** The recorded grade of a query, matched on the question and the query text, spacing and case free. */
export function recordedGrade(session: RecordedSession, questionId: string, sql: string): Grade | null {
  const hit = session.grades.find((g) => g.request.question_id === questionId && squash(g.request.sql) === squash(sql));
  return hit && hit.status === 200 ? hit.response : null;
}
