"use client";

import { useId } from "react";

import { loadIndex, loadRunJson, loadRunLines, loadSource } from "@/lib/data";
import { formatValue } from "@/lib/format";
import { devCurves } from "@/lib/grades";
import { heroMessage, panelLine, type PanelSummary } from "@/lib/rules";
import { ARM_LABEL, ARMS, type Arm, armRuns, paidOf } from "@/lib/runs";
import type { AlertRow, MonitorRow, Plan, Predictions } from "@/lib/types";
import { useLoad } from "@/lib/useLoad";

import { ChartDefs, fill, PATTERN } from "./Defs";
import { Figure } from "./Figure";
import { interpolate, linear, path, ticks } from "./scale";
import { useWidth } from "./useWidth";

export interface PanelData extends PanelSummary {
  run: string | null;
  paidSeries: [number, number][];
  confirmedSeries: [number, number][];
  firedStat: string | null;
}

interface HeroData {
  panels: PanelData[];
  budget: number;
  sql: string;
  anyRun: boolean;
  sources: string[];
}

async function loadHero(): Promise<HeroData> {
  const [index, predictions, plan] = await Promise.all([
    loadIndex(),
    loadSource<Predictions>("evaluation/predictions.yaml"),
    loadSource<Plan>("evaluation/plan.yaml"),
  ]);
  const byArm = armRuns(index);
  const runs = Object.values(byArm).filter((r): r is string => Boolean(r));
  const curves = runs.length ? await devCurves(index, runs) : { sql: "", rows: [] };
  const sources = new Set<string>(["evaluation/predictions.yaml"]);
  const panels: PanelData[] = [];
  for (const arm of ARMS) {
    const run = byArm[arm] ?? null;
    const prediction = predictions?.defects?.[arm] ?? null;
    const window = prediction?.window ? ([prediction.window[0], prediction.window[1]] as [number, number]) : null;
    let paidSeries: [number, number][] = [];
    let confirmedSeries: [number, number][] = [];
    let firedAt: number | null = null;
    let firedStat: string | null = null;
    if (run) {
      const entry = index.runs[run]!;
      for (const f of ["monitor", "alerts", "trainer"] as const) if (entry.files[f]) sources.add(entry.files[f]!);
      const monitor = await loadRunLines<MonitorRow>(entry, "monitor");
      paidSeries = monitor.map((r) => [Number(r.step), paidOf(r)] as [number, number | null]).filter((p): p is [number, number] => p[1] !== null);
      if (!paidSeries.length) {
        const trainer = await loadRunLines<MonitorRow>(entry, "trainer");
        paidSeries = trainer.map((r) => [Number(r.step), paidOf(r)] as [number, number | null]).filter((p): p is [number, number] => p[1] !== null);
      }
      paidSeries.sort((a, b) => a[0] - b[0]);
      confirmedSeries = curves.rows.filter((c) => c.run === run && c.step !== null).map((c) => [Number(c.step), Number(c.twin_confirmed)] as [number, number]);
      confirmedSeries.sort((a, b) => a[0] - b[0]);
      const alerts = ((await loadRunJson<AlertRow[]>(entry, "alerts")) ?? []).slice().sort((a, b) => a.step - b.step);
      const predicted = prediction?.detector ? alerts.find((a) => a.stat === prediction.detector) : undefined;
      const first = predicted ?? alerts[0];
      if (first) {
        firedAt = first.step;
        firedStat = first.stat;
      }
    }
    const lastConfirmed = confirmedSeries.at(-1) ?? null;
    const paidAtLast = lastConfirmed ? interpolate(paidSeries, lastConfirmed[0]) ?? paidSeries.at(-1)?.[1] ?? null : null;
    panels.push({
      arm,
      label: ARM_LABEL[arm],
      run,
      ran: Boolean(run),
      lastStep: lastConfirmed ? lastConfirmed[0] : null,
      paid: paidAtLast,
      confirmed: lastConfirmed ? lastConfirmed[1] : null,
      firedAt,
      firedStat,
      window,
      paidSeries,
      confirmedSeries,
    });
  }
  if (Object.keys(index.grades).length && runs.length) sources.add("results/grades/index.json");
  const stepBudget = Number((plan?.defect_study as { step_budget?: number } | undefined)?.step_budget ?? 0);
  const maxStep = Math.max(stepBudget, ...panels.flatMap((p) => [...p.paidSeries, ...p.confirmedSeries].map((s) => s[0])), ...panels.map((p) => p.window?.[1] ?? 0));
  return { panels, budget: maxStep || 1, sql: curves.sql, anyRun: runs.length > 0, sources: [...sources] };
}

/** The hero: one panel per run of the defect study, the reward paid against what the twins confirmed. */
export function PaidEarned() {
  const load = useLoad(loadHero, []);
  const data = load.value;
  const message = data ? heroMessage(data.panels) : null;
  const table = data
    ? {
        columns: ["arm", "run", "step", "reward paid", "twin confirmed on dev"],
        formats: ["text", "text", "int", "pct1", "pct1"],
        rows: data.panels.flatMap((p) =>
          p.confirmedSeries.length
            ? p.confirmedSeries.map(([step, c]) => [p.label, p.run ?? "", step, interpolate(p.paidSeries, step), c])
            : [[p.label, p.run ?? "not yet run", null, null, null]],
        ),
      }
    : null;
  return (
    <Figure
      testId="hero"
      message={message}
      subtitle="Each panel is one run from the same start, seed and prompts; only the reward differs. Red is the mean reward paid per training step; green is twin confirmed accuracy on dev at every saved checkpoint; the hatch between them is the gap; a band is the window the registered prediction named; a vertical rule is the step a monitor fired."
      sources={data?.sources ?? ["evaluation/predictions.yaml"]}
      sql={data?.anyRun ? data.sql : null}
      sqlNote="The green line is this query over the graded dev completions. The red line is read from each run's monitor.jsonl, one row per step, as the trainer logged it."
      table={table}
      state={load.status === "loading" ? "loading" : load.status === "error" ? "error" : "ready"}
      error={load.error?.message}
      skeleton="multiples"
      height={380}
    >
      {data ? <Multiples data={data} /> : null}
    </Figure>
  );
}

function Multiples({ data }: { data: HeroData }) {
  return (
    <div>
      <Legend panels={data.panels} />
      <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 mt-3">
        {data.panels.map((p) => (
          <Panel key={p.arm} panel={p} budget={data.budget} />
        ))}
      </div>
    </div>
  );
}

function Legend({ panels }: { panels: PanelData[] }) {
  return (
    <div className="text-sm text-ink2" data-testid="hero-legend">
      <p>
        <span className="font-semibold text-ink">Panels: </span>
        {panels.map((p, i) => (
          <span key={p.arm} data-legend-panel={p.arm}>
            {i > 0 ? ", " : null}
            {p.label}
          </span>
        ))}
        .
      </p>
      <ul className="flex flex-wrap gap-x-5 gap-y-1 mt-1" aria-label="How to read a panel">
        <li className="inline-flex items-center gap-1.5">
          <svg width="22" height="8" aria-hidden="true">
            <line x1="0" y1="4" x2="22" y2="4" stroke="var(--gh-reward)" strokeWidth="2" />
          </svg>
          reward paid (solid)
        </li>
        <li className="inline-flex items-center gap-1.5">
          <svg width="22" height="8" aria-hidden="true">
            <line x1="0" y1="4" x2="22" y2="4" stroke="var(--gh-checks)" strokeWidth="2.2" strokeDasharray="5 3" />
            <circle cx="11" cy="4" r="2.5" fill="var(--gh-checks)" />
          </svg>
          twin confirmed on dev (dashed, dots at checkpoints)
        </li>
        <li className="inline-flex items-center gap-1.5">
          <svg width="16" height="12" aria-hidden="true">
            <defs>
              <pattern id="legend-gap" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="5" stroke="var(--gh-reward)" strokeWidth="1.3" />
              </pattern>
            </defs>
            <rect width="16" height="12" fill="url(#legend-gap)" stroke="var(--gh-hairline)" />
          </svg>
          the gap
        </li>
        <li className="inline-flex items-center gap-1.5">
          <svg width="16" height="12" aria-hidden="true">
            <rect width="16" height="12" fill="var(--gh-seq-1)" opacity="0.5" />
          </svg>
          predicted window
        </li>
        <li className="inline-flex items-center gap-1.5">
          <svg width="10" height="12" aria-hidden="true">
            <line x1="5" y1="0" x2="5" y2="12" stroke="var(--gh-ink)" strokeWidth="1.5" strokeDasharray="2 2" />
          </svg>
          monitor fired
        </li>
      </ul>
    </div>
  );
}

function Panel({ panel, budget }: { panel: PanelData; budget: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(300);
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const height = 168;
  const m = { left: 36, right: 100, top: 14, bottom: 22 };
  const top = Math.max(1, ...panel.paidSeries.map((p) => p[1]));
  const x = linear([0, budget], [m.left, Math.max(m.left + 40, width - m.right)]);
  const y = linear([0, top], [height - m.bottom, m.top]);
  const drawn = panel.ran && (panel.paidSeries.length > 0 || panel.confirmedSeries.length > 0);
  const lastPaid = panel.paidSeries.at(-1);
  const lastConf = panel.confirmedSeries.at(-1);
  // The gap: between the paid line and the confirmed line, over the steps both cover.
  const span = panel.confirmedSeries.length > 1 ? [panel.confirmedSeries[0]![0], lastConf![0]] : null;
  const upper = span ? panel.paidSeries.filter(([s]) => s >= span[0] && s <= span[1]) : [];
  const gapPoints = upper.length > 1 ? [...upper.map(([s, v]) => [x(s), y(v)]), ...[...upper].reverse().map(([s]) => [x(s), y(interpolate(panel.confirmedSeries, s) ?? 0)])] : [];
  let paidLabelY = lastPaid ? y(lastPaid[1]) : 0;
  let confLabelY = lastConf ? y(lastConf[1]) : 0;
  if (lastPaid && lastConf && Math.abs(paidLabelY - confLabelY) < 13) {
    const mid = (paidLabelY + confLabelY) / 2;
    paidLabelY = lastPaid[1] >= lastConf[1] ? mid - 7 : mid + 7;
    confLabelY = lastPaid[1] >= lastConf[1] ? mid + 7 : mid - 7;
  }
  return (
    <figure className="border border-hairline bg-raised p-2 min-w-0" data-panel={panel.arm} data-drawn={drawn ? "true" : "false"} data-testid="hero-panel">
      <figcaption className="px-1 min-w-0">
        <span className="block font-semibold text-ink text-sm">{panel.label}</span>
        <span className="block text-xs text-ink2 font-mono truncate" title={panel.run ?? undefined}>
          {panel.run ?? "no run yet"}
        </span>
      </figcaption>
      <div ref={ref} className="w-full">
        <svg width={width} height={height} role="img" aria-label={`${panel.label}: ${panelLine(panel)}`} className="block">
          <ChartDefs uid={uid} />
          {panel.window ? (
            <g data-band="window">
              <rect x={x(panel.window[0])} y={m.top} width={Math.max(1, x(panel.window[1]) - x(panel.window[0]))} height={height - m.top - m.bottom} fill={fill(PATTERN.window, uid)} />
              <text x={x(panel.window[0]) + 3} y={m.top + 9} fontSize={10} className="fill-ink2">
                predicted
              </text>
            </g>
          ) : null}
          {ticks(0, top, 2).map((t) => (
            <g key={t}>
              <line x1={m.left} x2={x(budget)} y1={y(t)} y2={y(t)} stroke="var(--gh-hairline)" />
              <text x={m.left - 5} y={y(t) + 3.5} fontSize={10} textAnchor="end" className="fill-ink2 figure">
                {formatValue(t, "pct0")}
              </text>
            </g>
          ))}
          {[0, budget / 2, budget].map((t) => (
            <text key={t} x={x(t)} y={height - 6} fontSize={10} textAnchor="middle" className="fill-ink2 figure">
              {formatValue(t, "int")}
            </text>
          ))}
          {gapPoints.length ? <polygon points={gapPoints.map((p) => p.join(",")).join(" ")} fill={fill(PATTERN.gap, uid)} data-series="gap" /> : null}
          {panel.paidSeries.length ? (
            <path d={path(panel.paidSeries.map(([s, v]) => [x(s), y(v)]))} fill="none" stroke="var(--gh-reward)" strokeWidth={1.6} data-series="paid" />
          ) : null}
          {panel.confirmedSeries.length ? (
            <g data-series="confirmed">
              <path d={path(panel.confirmedSeries.map(([s, v]) => [x(s), y(v)]))} fill="none" stroke="var(--gh-checks)" strokeWidth={2.2} strokeDasharray="5 3" />
              {panel.confirmedSeries.map(([s, v]) => (
                <circle key={s} cx={x(s)} cy={y(v)} r={2.6} fill="var(--gh-checks)" />
              ))}
            </g>
          ) : null}
          {panel.firedAt !== null ? (
            <g data-rule="fired">
              <line x1={x(panel.firedAt)} x2={x(panel.firedAt)} y1={m.top} y2={height - m.bottom} stroke="var(--gh-ink)" strokeWidth={1.4} strokeDasharray="2 2" />
              <text x={x(panel.firedAt) + 3} y={height - m.bottom - 4} fontSize={10} className="fill-ink">
                fired {formatValue(panel.firedAt, "int")}
              </text>
            </g>
          ) : null}
          {lastPaid ? (
            <text x={x(lastPaid[0]) + 5} y={paidLabelY + 3.5} fontSize={10.5} fill="var(--gh-reward)" className="figure">
              paid {formatValue(lastPaid[1], "pct0")}
            </text>
          ) : null}
          {lastConf ? (
            <text x={x(lastConf[0]) + 5} y={confLabelY + 3.5} fontSize={10.5} fill="var(--gh-checks)" className="figure">
              confirmed {formatValue(lastConf[1], "pct0")}
            </text>
          ) : null}
          {!panel.ran ? (
            <text x={(m.left + x(budget)) / 2} y={height / 2} fontSize={13} textAnchor="middle" className="fill-ink2" data-testid="panel-not-run">
              not yet run
            </text>
          ) : null}
        </svg>
      </div>
      <p className="px-1 text-xs text-ink2 leading-snug" data-testid="panel-line">
        {panelLine(panel)}
      </p>
    </figure>
  );
}
