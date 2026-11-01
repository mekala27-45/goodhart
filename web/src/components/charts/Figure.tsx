"use client";

import { usePathname } from "next/navigation";
import { type ReactNode, useId, useState } from "react";

import { loadIndex, type DataIndex } from "@/lib/data";
import { formatValue, type Scalar } from "@/lib/format";
import { repoUrl } from "@/lib/site";
import { useLoad } from "@/lib/useLoad";

export interface TableData {
  columns: string[];
  formats: string[];
  rows: Scalar[][];
}

export type FigureState = "loading" | "ready" | "missing" | "error";
export type SkeletonShape = "lines" | "bars" | "dots" | "multiples" | "table";

/**
 * Every chart: its one message as the title (written by a rule from the rows it draws), a table
 * toggle and a show the SQL toggle beside it, the chart or its table or its SQL, and the files it
 * read with their digests. While it loads it is a skeleton in the chart's shape; when its data does
 * not exist yet it says so in words.
 */
export function Figure({
  message,
  subtitle,
  sources,
  sql,
  sqlNote,
  table,
  state,
  error,
  missing,
  skeleton,
  height = 260,
  testId,
  children,
}: {
  message: string | null;
  subtitle?: ReactNode;
  sources: string[];
  sql?: string | null;
  sqlNote?: string;
  table?: TableData | null;
  state: FigureState;
  error?: string | null;
  missing?: ReactNode;
  skeleton: SkeletonShape;
  height?: number;
  testId?: string;
  children?: ReactNode;
}) {
  const [view, setView] = useState<"chart" | "table" | "sql">("chart");
  const id = useId();
  const toggle = (next: "table" | "sql") => setView((v) => (v === next ? "chart" : next));
  const ready = state === "ready";
  return (
    <figure className="sheet p-4 sm:p-5 min-w-0" aria-labelledby={id} data-testid={testId} data-state={state} data-view={view}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h3 id={id} className="font-sans text-[1.0625rem] font-bold text-ink leading-snug max-w-[60ch]" data-testid="chart-message">
          {state === "loading" ? <span className="skeleton inline-block h-4 w-72 max-w-full align-middle" aria-hidden="true" /> : message}
          {state === "loading" ? <span className="sr-only">Loading</span> : null}
        </h3>
        {ready ? (
          <div className="no-print flex gap-2 shrink-0">
            <ToggleButton active={view === "table"} onClick={() => toggle("table")} testId="table-toggle">
              {view === "table" ? "Show the chart" : "Show the table"}
            </ToggleButton>
            {sql ? (
              <ToggleButton active={view === "sql"} onClick={() => toggle("sql")} testId="sql-toggle">
                {view === "sql" ? "Hide the SQL" : "Show the SQL"}
              </ToggleButton>
            ) : null}
          </div>
        ) : null}
      </div>
      {subtitle ? <p className="text-sm text-ink2 mt-1 mb-3 measure">{subtitle}</p> : <div className="mb-3" />}
      {state === "loading" ? <ChartSkeleton shape={skeleton} height={height} /> : null}
      {state === "missing" ? missing : null}
      {state === "error" ? (
        <p className="text-sm text-ink" role="alert">
          The data for this chart did not load: {error ?? "the server did not answer"}.
        </p>
      ) : null}
      {ready ? (
        <>
          {view === "table" ? <DataTable table={table ?? null} /> : null}
          {view === "sql" && sql ? (
            <div>
              {sqlNote ? <p className="text-xs text-ink2 mb-2 measure">{sqlNote}</p> : null}
              <pre className="sql" data-testid="sql">
                {sql}
              </pre>
            </div>
          ) : null}
          <div hidden={view !== "chart"}>{children}</div>
        </>
      ) : null}
      <SourceLine sources={sources} />
    </figure>
  );
}

function ToggleButton({ active, onClick, testId, children }: { active: boolean; onClick: () => void; testId: string; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      data-testid={testId}
      className="text-xs px-2 py-1 border border-hairline rounded-sm text-ink2 bg-raised hover:text-ink hover:border-control aria-pressed:text-ink aria-pressed:border-control"
    >
      {children}
    </button>
  );
}

/** The files a chart read, each with the first characters of its digest and where it came from. A
 * chart that read the same file from many runs names the file once, with the number of runs. */
export function SourceLine({ sources }: { sources: string[] }) {
  const index = useLoad<DataIndex>(loadIndex, []);
  if (!sources.length) return null;
  const perRun = new Map<string, string[]>();
  const single: string[] = [];
  for (const s of sources) {
    const m = /^results\/runs\/([^/]+)\/(.+)$/.exec(s);
    if (m) perRun.set(m[2]!, [...(perRun.get(m[2]!) ?? []), s]);
    else single.push(s);
  }
  const grouped = [...perRun.entries()].filter(([, list]) => list.length > 2);
  const listed = [...single, ...[...perRun.entries()].filter(([, list]) => list.length <= 2).flatMap(([, list]) => list)];
  const origins = sources.map((s) => index.value?.sources[s]?.origin).filter(Boolean);
  const fixture = origins.includes("fixture");
  const allFixture = fixture && origins.every((o) => o === "fixture");
  return (
    <p className="mt-3 text-xs text-ink2 leading-snug" data-testid="source-line">
      Read from{" "}
      {listed.map((s, i) => {
        const entry = index.value?.sources[s];
        return (
          <span key={s}>
            {i > 0 ? ", " : null}
            <a href={repoUrl(s)} className="font-mono">
              {s}
            </a>
            {entry ? <span className="font-mono"> ({entry.sha256.slice(0, 8)})</span> : null}
          </span>
        );
      })}
      {grouped.map(([file, list], i) => (
        <span key={file}>
          {listed.length || i > 0 ? ", " : null}
          the <span className="font-mono">{file}</span> of {formatValue(list.length, "int")} runs under{" "}
          <a href={repoUrl("results/runs")} className="font-mono">
            results/runs
          </a>
        </span>
      ))}
      .
      {fixture ? (
        <strong className="text-ink">{allFixture ? " These files are end to end test fixtures, not results." : " Some of these files are end to end test fixtures, not results."}</strong>
      ) : null}
    </p>
  );
}

/** The pages where tables set in the condensed face belong; everywhere else a table is set in the body face. */
const DENSE_PAGES = ["/methods", "/verifier", "/sizing"];

export function DataTable({ table, max = 300, caption, dense }: { table: TableData | null; max?: number; caption?: string; dense?: boolean }) {
  const path = usePathname() ?? "";
  if (!table) return <p className="text-sm text-ink2">There is no table for this chart.</p>;
  dense ??= DENSE_PAGES.some((p) => path.startsWith(p));
  const rows = table.rows.slice(0, max);
  return (
    <div className="overflow-x-auto max-h-[460px] overflow-y-auto" role="region" aria-label={caption ?? "The chart as a table"} tabIndex={0}>
      <table className={`data-table${dense ? " dense" : ""}`} data-testid="data-table">
        {caption ? <caption className="text-left text-sm text-ink2 pb-2">{caption}</caption> : null}
        <thead>
          <tr>
            {table.columns.map((c, i) => (
              <th key={`${c}-${i}`} scope="col" className={table.formats[i] === "text" ? "" : "num"}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, i) => (
                <td key={i} className={table.formats[i] === "text" ? "" : "num"}>
                  {formatValue(cell, table.formats[i] ?? "text")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {table.rows.length > max ? (
        <p className="text-xs text-ink2 mt-2">
          The first {formatValue(max, "int")} of {formatValue(table.rows.length, "int")} rows.
        </p>
      ) : null}
    </div>
  );
}

/** A loading state in the shape of the chart it stands in for. */
export function ChartSkeleton({ shape, height = 260 }: { shape: SkeletonShape; height?: number }) {
  return (
    <div aria-busy="true" role="img" aria-label="Loading the chart" style={{ minHeight: height }} data-skeleton={shape}>
      {shape === "lines" ? (
        <div className="relative w-full" style={{ height }}>
          {[0.15, 0.4, 0.65, 0.9].map((t) => (
            <div key={t} className="absolute left-8 right-2 border-t border-hairline" style={{ top: `${t * 100}%` }} />
          ))}
          <div className="skeleton absolute left-8 right-2 top-[30%] h-1" />
          <div className="skeleton absolute left-8 right-2 top-[62%] h-1" />
        </div>
      ) : null}
      {shape === "multiples" ? (
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 7 }, (_, i) => (
            <div key={i} className="border border-hairline p-2" style={{ height: 170 }}>
              <div className="skeleton h-3 w-24 mb-3" />
              <div className="skeleton h-1 w-full mt-10" />
              <div className="skeleton h-1 w-4/5 mt-8" />
            </div>
          ))}
        </div>
      ) : null}
      {shape === "bars" || shape === "table" ? (
        <div className="space-y-3 pt-1">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="skeleton h-3 w-24 shrink-0" />
              <div className="skeleton h-3" style={{ width: `${30 + ((i * 17) % 50)}%` }} />
            </div>
          ))}
        </div>
      ) : null}
      {shape === "dots" ? (
        <div className="space-y-4 pt-2">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="skeleton h-3 w-20 shrink-0" />
              <div className="relative flex-1 h-3">
                <div className="absolute inset-y-1 left-0 right-0 border-t border-hairline" />
                <div className="skeleton absolute h-3 w-3 rounded-full" style={{ left: `${25 + ((i * 13) % 55)}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : null}
      <span className="sr-only">Loading the chart</span>
    </div>
  );
}
