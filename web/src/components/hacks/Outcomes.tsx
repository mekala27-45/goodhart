"use client";

import { useId } from "react";

import { ChartDefs, fill, PATTERN } from "@/components/charts/Defs";
import { Figure } from "@/components/charts/Figure";
import { linear } from "@/components/charts/scale";
import { useWidth } from "@/components/charts/useWidth";
import { formatValue } from "@/lib/format";
import { loadHacks, type Outcome } from "@/lib/hacks";
import { countWord, listWords, plural, sentence } from "@/lib/messages";
import { useLoad } from "@/lib/useLoad";

/** The title of the predictions chart, by rule. */
export function predictionsMessage(outcomes: Outcome[]): string {
  const judged = outcomes.filter((o) => o.kind !== "honest" && o.held !== null);
  const ran = outcomes.filter((o) => o.kind !== "honest" && o.run);
  if (!ran.length) return "No defect arm or control has run yet; every prediction below was registered before any of them did.";
  if (!judged.length) return sentence(`${plural(ran.length, "arm")} ran, and none has the graded checkpoints its prediction needs yet.`);
  const held = judged.filter((o) => o.held).map((o) => o.label);
  const not = judged.filter((o) => !o.held).map((o) => o.label);
  if (!not.length) return `Every prediction that can be judged held: ${listWords(held)}.`;
  if (!held.length) return `No prediction that can be judged held: ${listWords(not)} did not.`;
  return sentence(`${plural(held.length, "prediction")} held (${listWords(held)}) and ${countWord(not.length)} did not (${listWords(not)}).`);
}

const fmtPts = (v: number | null) => (v === null ? "not measured" : `${formatValue(v / 100, "apts1")}`);

/** One row per defect and control: the predicted window, the step the detector fired, and the verdict. */
export function Outcomes() {
  const load = useLoad(loadHacks, []);
  const d = load.value;
  const rows = d?.outcomes.filter((o) => o.kind !== "honest") ?? [];
  return (
    <Figure
      testId="predictions"
      message={d ? predictionsMessage(d.outcomes) : null}
      subtitle="The hatched band is the window the prediction named for its detector; the diamond is the step the monitor fired. A defect's prediction held when its detector fired inside the window and, by its last checkpoint, the twins confirmed at least the stated points less than the honest arm on dev."
      sources={["evaluation/predictions.yaml", "config/monitor.yaml", "results/grades/index.json"]}
      sql={d?.sql || null}
      sqlNote="The twin confirmed accuracy and over refusal at each arm's last checkpoint come from this query over the graded dev completions; the firing step is the first row of the arm's alerts.json for the predicted detector."
      table={{
        columns: ["arm", "detector", "window from", "window to", "fired at", "twin gap, points", "over refusal, points", "share of honest gain", "verdict"],
        formats: ["text", "text", "int", "int", "int", "float1", "float1", "pct1", "text"],
        rows: rows.map((o) => [o.label, o.prediction?.detector ?? "", o.prediction?.window?.[0] ?? null, o.prediction?.window?.[1] ?? null, o.firedAt, o.gapPoints, o.overRefusalPoints, o.shareOfGain, o.word]),
      }}
      state={load.status === "loading" ? "loading" : load.status === "error" ? "error" : "ready"}
      error={load.error?.message}
      skeleton="bars"
      height={300}
    >
      {d ? <Timeline outcomes={rows} budget={Math.max(200, ...rows.map((o) => o.prediction?.window?.[1] ?? 0), ...rows.map((o) => o.lastStep ?? 0))} /> : null}
    </Figure>
  );
}

function Timeline({ outcomes, budget }: { outcomes: Outcome[]; budget: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const narrow = width < 600;
  const labelW = narrow ? 0 : 130;
  const verdictW = narrow ? 0 : 110;
  const rowH = narrow ? 46 : 30;
  const top = 18;
  const height = top + outcomes.length * rowH + 6;
  const x = linear([0, budget], [labelW + 4, width - verdictW - 8]);
  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={height} role="img" aria-label="Predicted windows and firing steps, one row per arm" className="block">
        <ChartDefs uid={uid} />
        {[0, budget / 2, budget].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 4} y2={height - 4} stroke="var(--gh-hairline)" />
            <text x={x(t)} y={11} fontSize={10} textAnchor="middle" className="fill-ink2 figure">
              step {formatValue(t, "int")}
            </text>
          </g>
        ))}
        {outcomes.map((o, i) => {
          const y0 = top + i * rowH;
          const mid = y0 + (narrow ? 30 : rowH / 2);
          const w = o.prediction?.window;
          return (
            <g key={o.arm} data-arm={o.arm} data-verdict={o.word}>
              <text x={0} y={narrow ? y0 + 12 : mid + 4} fontSize={12.5} fontWeight={600} className="fill-ink">
                {o.label}
                {narrow ? <tspan className="fill-ink2" fontWeight={400}>{`, ${o.word}`}</tspan> : null}
              </text>
              <line x1={x(0)} x2={x(budget)} y1={mid} y2={mid} stroke="var(--gh-hairline)" />
              {w ? <rect x={x(w[0])} y={mid - 7} width={Math.max(1, x(w[1]) - x(w[0]))} height={14} fill={fill(PATTERN.window, uid)} stroke="var(--gh-seq-2)" strokeWidth={0.6} /> : null}
              {o.firedAt !== null ? (
                <g data-fired={o.firedAt}>
                  <rect x={x(o.firedAt) - 5} y={mid - 5} width={10} height={10} transform={`rotate(45 ${x(o.firedAt)} ${mid})`} fill="var(--gh-reward)" />
                  <text x={x(o.firedAt) + 9} y={mid - 6} fontSize={10.5} className="fill-ink figure">
                    fired {formatValue(o.firedAt, "int")}
                  </text>
                </g>
              ) : (
                <text x={x(0) + 6} y={mid - 6} fontSize={10.5} className="fill-ink2">
                  {o.run ? "did not fire" : "not yet run"}
                </text>
              )}
              {!narrow ? (
                <text x={width - verdictW + 4} y={mid + 4} fontSize={12.5} fontWeight={600} className={o.word === "held" ? "fill-checks" : o.word === "did not hold" ? "fill-reward" : "fill-ink2"}>
                  {o.word}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Each arm's prediction in words, beside its outcome. */
export function PredictionCards() {
  const load = useLoad(loadHacks, []);
  const d = load.value;
  if (!d) return null;
  return (
    <ol className="grid gap-3 sm:grid-cols-2" data-testid="prediction-cards">
      {d.outcomes
        .filter((o) => o.kind !== "honest")
        .map((o) => {
          const p = o.prediction;
          return (
            <li key={o.arm} className="sheet p-4" data-arm={o.arm}>
              <p className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-semibold">{o.label}</span>
                <span className={`text-sm font-semibold ${o.word === "held" ? "text-checks" : o.word === "did not hold" ? "text-reward" : "text-ink2"}`} data-testid="verdict">
                  {o.word}
                </span>
              </p>
              {p ? (
                <p className="text-sm mt-1">
                  <span className="text-ink2">Predicted: </span>
                  {p.detector ? (
                    <>
                      <span className="font-mono text-[0.8125rem]">{p.detector}</span> fires between steps {formatValue(p.window?.[0] ?? null, "int")} and {formatValue(p.window?.[1] ?? null, "int")}
                      {p.twin_gap_points_at_least !== undefined ? `, and the twins confirm at least ${formatValue(p.twin_gap_points_at_least / 100, "apts1")} less than the honest arm` : ""}
                      {p.over_refusal_points_at_least !== undefined ? `, and over refusal rises at least ${formatValue(p.over_refusal_points_at_least / 100, "apts1")} above the honest arm` : ""}.
                    </>
                  ) : p.share_of_honest_gain_below !== undefined ? (
                    <>reaches less than {formatValue(p.share_of_honest_gain_below, "pct0")} of the honest arm&apos;s gain.</>
                  ) : null}
                </p>
              ) : (
                <p className="text-sm mt-1 text-ink2">No prediction is registered for this arm.</p>
              )}
              {p?.why ? <p className="text-sm text-ink2 mt-1">Why: {p.why}.</p> : null}
              <p className="text-sm mt-2">
                <span className="text-ink2">Outcome: </span>
                {o.run ? (
                  <>
                    {o.firedAt !== null ? `${o.firedStat} fired at step ${formatValue(o.firedAt, "int")}` : "no monitor fired"}
                    {o.gapPoints !== null ? `; twin gap ${fmtPts(o.gapPoints)} at step ${formatValue(o.lastStep, "int")}` : ""}
                    {o.overRefusalPoints !== null && p?.over_refusal_points_at_least !== undefined ? `; over refusal ${fmtPts(o.overRefusalPoints)} above the honest arm` : ""}
                    {o.shareOfGain !== null ? `; ${formatValue(o.shareOfGain, "pct1")} of the honest gain` : ""}. {o.reason}
                  </>
                ) : (
                  "not yet run."
                )}
              </p>
            </li>
          );
        })}
    </ol>
  );
}
