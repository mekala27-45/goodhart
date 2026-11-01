"use client";

import { useEffect, useMemo, useState } from "react";

import { DataTable } from "@/components/charts/Figure";
import { Script } from "@/components/Script";
import {
  apiSource,
  type ApiQuestion,
  type Grade,
  type Health,
  liveGrade,
  liveQuestions,
  recordedGrade,
  recordedSession,
  type RecordedSession,
  type Source,
  TRIES,
  type Try,
} from "@/lib/api";
import { loadIndex, loadSource, loadUrl, type DataIndex } from "@/lib/data";
import { formatValue } from "@/lib/format";
import { marks as gradeMarks, type MarkRow } from "@/lib/grades";
import { flagsOf, marksFromGrade, marksFromRow } from "@/lib/marks";
import { methodName, methodOfRun } from "@/lib/runs";
import { API } from "@/lib/site";
import type { PublicQuestion, ScriptSet } from "@/lib/types";
import { useLoad } from "@/lib/useLoad";

const SET_WORDS: Record<string, string> = { readback_test: "readback test", test: "generated test", template_test: "template test" };

interface DeskState {
  source: Source;
  health: Health | null;
  session: RecordedSession | null;
  questions: ApiQuestion[];
  questionsFrom: string;
}

async function loadDesk(): Promise<DeskState> {
  const [{ source, health }, session] = await Promise.all([apiSource(), recordedSession().catch(() => null)]);
  if (source === "live") {
    try {
      const list = await liveQuestions();
      return { source, health, session, questions: list.questions, questionsFrom: `${API}/v1/questions` };
    } catch {
      // The API answered /health and then failed the list: fall through to the published questions.
    }
  }
  const published = (await loadSource<PublicQuestion[]>("data/questions/readback_test.jsonl")) ?? [];
  const recorded = session?.questions.questions ?? [];
  const seen = new Set(recorded.map((q) => q.id));
  const questions = [...recorded, ...published.filter((q) => !seen.has(q.id)).map((q) => ({ id: q.id, set: q.set, question: q.question, family: q.family, warehouse: q.warehouse }))];
  return { source, health, session, questions, questionsFrom: source === "live" ? "the published readback test questions" : "the recorded session and the published readback test questions" };
}

export function Desk() {
  const desk = useLoad(loadDesk, []);
  const index = useLoad(loadIndex, []);
  const [selected, setSelected] = useState<string | null>(null);

  const d = desk.value;
  const triesByQuestion = useMemo(() => {
    const map = new Map<string, Try[]>();
    for (const t of d?.session?.tries ?? []) map.set(t.question_id, [...(map.get(t.question_id) ?? []), t]);
    return map;
  }, [d]);

  useEffect(() => {
    if (!d || selected) return;
    const withTries = d.questions.find((q) => triesByQuestion.has(q.id));
    setSelected(withTries?.id ?? d.questions[0]?.id ?? null);
  }, [d, selected, triesByQuestion]);

  if (desk.status === "loading") {
    return (
      <div className="space-y-4" aria-busy="true" data-testid="desk-loading">
        <p className="text-sm text-ink2">Asking the grader whether it is awake: two probes of /health, a few seconds apart.</p>
        <div className="script h-48 skeleton" />
      </div>
    );
  }
  if (desk.status === "error") return <p role="alert">The desk did not load: {desk.error.message}.</p>;
  const question = d!.questions.find((q) => q.id === selected) ?? null;
  return (
    <div className="space-y-8">
      <SourceNote state={d!} />
      <div>
        <label htmlFor="question" className="block text-sm font-semibold mb-1">
          The question
        </label>
        <select
          id="question"
          data-testid="question-picker"
          className="w-full max-w-full sheet px-2 py-2 text-sm text-ink bg-raised"
          value={selected ?? ""}
          onChange={(e) => setSelected(e.target.value)}
        >
          {Object.entries(groupBySet(d!.questions)).map(([set, qs]) => (
            <optgroup key={set} label={SET_WORDS[set] ?? set}>
              {qs.map((q) => (
                <option key={q.id} value={q.id}>
                  {triesByQuestion.has(q.id) ? "(with tries) " : ""}
                  {q.question}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <p className="text-xs text-ink2 mt-1">
          {formatValue(d!.questions.length, "int")} held-out questions, from {d!.questionsFrom}. The gold answer of a question never reaches this page.
        </p>
      </div>
      {question ? (
        <>
          <Bench key={question.id} question={question} state={d!} tries={triesByQuestion.get(question.id) ?? []} />
          <MethodScripts key={`m-${question.id}`} question={question} index={index.value} />
        </>
      ) : null}
    </div>
  );
}

function groupBySet(questions: ApiQuestion[]): Record<string, ApiQuestion[]> {
  const out: Record<string, ApiQuestion[]> = {};
  for (const q of questions) (out[q.set] ??= []).push(q);
  return out;
}

function SourceNote({ state }: { state: DeskState }) {
  if (state.source === "live") {
    return (
      <p className="sheet px-4 py-3 text-sm" data-testid="desk-source" data-source="live">
        <span className="font-semibold text-checks">Live.</span> The grader at <span className="font-mono">{API}</span> answered its health check
        {state.health?.version ? `, version ${state.health.version}` : ""}
        {state.health?.verifier ? `, verifier ${state.health.verifier}` : ""}
        {state.health?.kept_twins ? `, ${formatValue(state.health.kept_twins.length, "int")} kept twins` : ""}. Every mark below is graded now, on the original and on the
        twins, and written to the grade log before it comes back.
      </p>
    );
  }
  const s = state.session;
  return (
    <div className="sheet px-4 py-3 text-sm space-y-1" data-testid="desk-source" data-source="recorded">
      <p>
        <span className="font-semibold text-reward">Recorded.</span> The grader did not answer two probes of its health check, so this desk is playing a recorded
        session: nothing below is graded now{s?.recorded_at ? `, and every mark was recorded on ${s.recorded_at}` : ""}.
      </p>
      {s?.placeholder ? (
        <p className="text-ink2" data-testid="placeholder-note">
          This recording is a placeholder with the exact shape of the API&apos;s answers: {s.recorded_from}. It is replaced by a recording of the live API.
        </p>
      ) : s ? (
        <p className="text-ink2">Recorded from {s.recorded_from}.</p>
      ) : (
        <p className="text-ink2">The recorded session did not load either, so only the published questions are shown.</p>
      )}
    </div>
  );
}

interface Marked {
  sql: string;
  label: string;
  grade: Grade | null;
  error: string | null;
}

function Bench({ question, state, tries }: { question: ApiQuestion; state: DeskState; tries: Try[] }) {
  const [sql, setSql] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Marked | null>(null);
  const [tried, setTried] = useState<Record<string, Grade | null>>({});

  async function mark(text: string, label: string): Promise<Grade | null> {
    if (!text.trim()) return null;
    setBusy(true);
    try {
      if (state.source === "live") {
        const { grade, error } = await liveGrade(question.id, text);
        setResult({ sql: text, label, grade, error });
        return grade;
      }
      const grade = state.session ? recordedGrade(state.session, question.id, text) : null;
      setResult({
        sql: text,
        label,
        grade,
        error: grade ? null : "The grader is asleep, and the recording has no grade for this query. The tries beside the editor were recorded; pick one of them.",
      });
      return grade;
    } catch (err) {
      setResult({ sql: text, label, grade: null, error: `the grader did not answer: ${(err as Error).message}` });
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function runTry(t: Try) {
    setSql(t.sql);
    const g = await mark(t.sql, t.label);
    setTried((prev) => ({ ...prev, [t.label]: g }));
  }

  const ordered = TRIES.map((label) => tries.find((t) => t.label === label)).filter((t): t is Try => Boolean(t));
  return (
    <section aria-labelledby="bench-title" className="space-y-4">
      <h2 id="bench-title" className="text-[1.4rem]">
        Write your own answer
      </h2>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <form
          className="space-y-2 min-w-0"
          onSubmit={(e) => {
            e.preventDefault();
            void mark(sql, "your query");
          }}
        >
          <p className="font-display text-[1.0625rem]">{question.question}</p>
          <p className="text-xs text-ink2 font-mono">
            {question.id} · {question.family} · {question.warehouse} · {SET_WORDS[question.set] ?? question.set}
          </p>
          <label htmlFor="sql" className="block text-sm font-semibold mt-2">
            Your answer: one SQL query, or REFUSE with a reason code
          </label>
          <textarea
            id="sql"
            data-testid="sql-editor"
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            rows={7}
            spellCheck={false}
            placeholder={"SELECT ... FROM doorstep.mart_metro_year WHERE ...\n\nor\n\nREFUSE FORECAST"}
            className="w-full script answer-ink text-[0.875rem] px-3 py-2"
          />
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={busy || !sql.trim()}
              data-testid="mark"
              className="px-4 py-1.5 border border-ink rounded-sm bg-ink text-raised font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {busy ? "Marking" : "Mark it"}
            </button>
            <span className="text-xs text-ink2">{state.source === "live" ? "Graded live by the API." : "Graded from the recording."}</span>
          </div>
        </form>
        <aside aria-label="Five tries" className="space-y-2" data-testid="tries">
          <p className="text-sm font-semibold">Five tries</p>
          {ordered.length ? (
            <ul className="space-y-2">
              {ordered.map((t) => {
                const g = tried[t.label] ?? (state.session ? recordedGrade(state.session, question.id, t.sql) : null);
                return (
                  <li key={t.label}>
                    <button
                      type="button"
                      onClick={() => void runTry(t)}
                      disabled={busy}
                      data-testid="try"
                      data-try={t.label}
                      className="w-full text-left sheet px-3 py-2 hover:border-control disabled:opacity-60"
                    >
                      <span className="block text-sm font-semibold text-ink">{t.label}</span>
                      {g ? (
                        <span className="block mt-0.5">
                          <span className="mark mark-reward text-[0.8rem]">honest {formatValue(g.pays.honest ?? null, "float2")}</span>{" "}
                          <span className="mark mark-checks text-[0.8rem]">
                            twins {g.checks.twins_passed} of {g.checks.twins_total}
                          </span>
                        </span>
                      ) : (
                        <span className="block text-xs text-ink2 mt-0.5">not marked yet</span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="text-sm text-ink2">
              The five tries (print the number, drop the filter, return the whole table, refuse, write it properly) are written for the questions marked
              &ldquo;with tries&rdquo; in the list; pick one of those.
            </p>
          )}
        </aside>
      </div>
      {result ? <GradeView question={question} result={result} source={state.source} /> : null}
    </section>
  );
}

function GradeView({ question, result, source }: { question: ApiQuestion; result: Marked; source: Source }) {
  const g = result.grade;
  if (!g) {
    return (
      <p className="sheet px-4 py-3 text-sm" role="alert" data-testid="grade-error">
        {result.error}
      </p>
    );
  }
  return (
    <div className="space-y-3" data-testid="grade" data-source={source}>
      <Script
        testId="grade-script"
        question={question.question}
        meta={`${question.id} · ${g.kind} · ${source === "live" ? `grade ${g.grade_id ?? "not logged"}` : "recorded"}`}
        who={`${result.label}, marked ${source === "live" ? "live" : "from the recording"}`}
        answer={result.sql}
        marks={marksFromGrade(g)}
        flags={flagsOf(g.flags)}
        note={g.error ? `The warehouse refused or failed the query: ${g.error}` : undefined}
      />
      {g.executed ? (
        <div className="sheet p-3">
          <p className="text-sm font-semibold mb-2">
            The result on the original warehouse: {formatValue(g.row_count, "int")} {g.row_count === 1 ? "row" : "rows"}
            {g.truncated ? ", the first shown" : ""}
          </p>
          <DataTable
            caption="Result rows"
            table={{
              columns: g.columns,
              formats: g.columns.map((_, i) => (g.rows.some((r) => typeof r[i] === "number") ? "float2" : "text")),
              rows: g.rows.map((r) => r.map((c) => (c === null || typeof c === "number" || typeof c === "string" ? c : String(c)))),
            }}
          />
        </div>
      ) : null}
    </div>
  );
}

interface MethodAnswer {
  run: string;
  label: string;
  text: string;
  checkpoint: string;
  row: MarkRow | null;
}

async function loadMethodAnswers(question: ApiQuestion, index: DataIndex | null): Promise<{ answers: MethodAnswer[]; graded: boolean } | null> {
  if (!index) return null;
  const entry = index.scripts[question.set];
  if (!entry) return null;
  const set = await loadUrl<ScriptSet>(entry.url);
  const item = set.items[question.id];
  if (!item) return null;
  const answers = item.answers.filter((a) => index.runs[a.run] && methodOfRun(index.runs[a.run]!));
  const requests = [...new Set(answers.map((a) => a.request))];
  let rows: MarkRow[] = [];
  let graded = false;
  try {
    rows = (await gradeMarks(index, question.set, [question.id], requests)).rows;
    graded = rows.length > 0;
  } catch {
    graded = false;
  }
  return {
    graded,
    answers: answers
      .map((a) => {
        const code = methodOfRun(index.runs[a.run]!)!;
        const row = rows.find((r) => r.request === a.request && r.checkpoint === a.checkpoint && r.sample === a.sample && Number(r.temperature) === Number(a.temperature)) ?? null;
        return { run: a.run, label: `${code} ${methodName(code)}`, text: a.text, checkpoint: a.checkpoint, row };
      })
      .sort((x, y) => x.label.localeCompare(y.label)),
  };
}

function MethodScripts({ question, index }: { question: ApiQuestion; index: DataIndex | null }) {
  const load = useLoad(() => loadMethodAnswers(question, index), [question.id, index]);
  return (
    <section aria-labelledby="methods-title" className="space-y-3">
      <h2 id="methods-title" className="text-[1.4rem]">
        How each method answered
      </h2>
      {load.status === "loading" ? (
        <div className="script h-40 skeleton" aria-busy="true" />
      ) : !load.value || !load.value.answers.length ? (
        <p className="sheet px-4 py-3 text-sm" data-testid="no-scripts">
          Not yet run: no method&apos;s answer to this question has been exported and graded. The scripts appear here once the sampling runs are promoted into
          results/runs and graded into results/grades.
        </p>
      ) : (
        <div className="space-y-3">
          {load.value.answers.map((a) => (
            <Script
              key={a.run}
              question={question.question}
              meta={`${question.id} · ${a.run} · ${a.checkpoint}`}
              who={`${a.label}, ${a.checkpoint}`}
              answer={a.text}
              marks={a.row ? marksFromRow(a.row, "honest") : []}
              flags={a.row ? flagsOf(a.row) : []}
              note={a.row ? undefined : "This answer has no grade in the published grades."}
            />
          ))}
        </div>
      )}
    </section>
  );
}
