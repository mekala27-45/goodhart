"use client";

import { useState } from "react";

import { CoverageChart } from "@/components/charts/Coverage";
import { DotRows } from "@/components/charts/DotRows";
import { DataTable, Figure } from "@/components/charts/Figure";
import { NotYetRun } from "@/components/NotYetRun";
import { loadIndex } from "@/lib/data";
import { formatValue } from "@/lib/format";
import { coverage } from "@/lib/grades";
import { differenceFor, loadManifest, loadMethods, type MethodScore } from "@/lib/methods";
import { compare, verdict } from "@/lib/messages";
import { KS } from "@/lib/passk";
import { coverageMessage, methodsMessage } from "@/lib/rules";
import { methodRuns, METHODS, type MethodCode } from "@/lib/runs";
import type { Hypotheses, Plan } from "@/lib/types";
import { useLoad } from "@/lib/useLoad";
import { stateOf, useSource } from "@/lib/useSource";

export const HELD_OUT: { key: string; label: string }[] = [
  { key: "readback_test", label: "readback test" },
  { key: "test", label: "generated test" },
  { key: "template_test", label: "template test" },
];

function SetPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div role="group" aria-label="Held-out set" className="flex flex-wrap gap-2">
      {HELD_OUT.map((s) => (
        <button
          key={s.key}
          type="button"
          aria-pressed={value === s.key}
          onClick={() => onChange(s.key)}
          className="text-sm px-2.5 py-1 border border-hairline rounded-sm bg-raised text-ink2 aria-pressed:text-ink aria-pressed:border-ink aria-pressed:font-semibold"
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}

function rows(methods: MethodScore[], floor: MethodScore) {
  return [
    ...methods.map((m) => ({
      key: m.code,
      label: m.label,
      color: `var(--gh-${m.token})`,
      point: m.score?.twin_confirmed ?? null,
      second: m.score?.execution ?? null,
      low: m.low,
      high: m.high,
      bar: m.gpuHours,
      barNote: m.code === "M0" ? "no training" : undefined,
    })),
    { key: "floor", label: "the floor", color: "var(--gh-control)", point: floor.score?.twin_confirmed ?? null, floor: true, bar: null, barNote: "no model, no GPU" },
  ];
}

/** Twin confirmed accuracy with intervals, every method on one held-out set at a time, the compute beside each. */
export function MethodsChart() {
  const [set, setSet] = useState("readback_test");
  const load = useLoad(() => loadMethods(set), [set]);
  const d = load.value;
  const label = HELD_OUT.find((s) => s.key === set)?.label ?? set;
  const diff = d && d.champion ? differenceFor(d.manifest, d.champion, "M0", set, "twin_confirmed_accuracy") : null;
  const any = Boolean(d && (d.methods.some((m) => m.score) || d.floor.score));
  return (
    <div className="space-y-3">
      <SetPicker value={set} onChange={setSet} />
      <Figure
        testId="methods-chart"
        message={d ? methodsMessage(d.methods.map((m) => ({ code: m.code, label: m.label, point: m.score?.twin_confirmed ?? null })), d.champion, diff, label) : null}
        subtitle={`Twin confirmed accuracy on the ${label} set, greedy where a greedy answer exists. Intervals are the repository's cluster bootstrap over question templates; the compute bar is the GPU hours the method's training runs used, from the run ledger.`}
        sources={["results/grades/index.json", "results/manifest.json"]}
        sql={d?.sql || null}
        table={
          d
            ? {
                columns: ["method", "twin confirmed", "low", "high", "execution", "over refusal", "refusal accuracy", "tokens per answer", "GPU hours"],
                formats: ["text", "pct1", "pct1", "pct1", "pct1", "pct1", "pct1", "float1", "hours2"],
                rows: [...d.methods, d.floor].map((m) => [m.label, m.score?.twin_confirmed ?? null, m.low, m.high, m.score?.execution ?? null, m.score?.over_refusal ?? null, m.score?.refusal_accuracy ?? null, m.score?.tokens ?? null, m.gpuHours]),
              }
            : null
        }
        state={stateOf(load, any)}
        error={load.error?.message}
        skeleton="dots"
        height={240}
        missing={<NotYetRun what={`Twin confirmed accuracy with its interval for every method on the ${label} set, the floor hatched, the base as the reference line and the compute beside each row.`} reads={["results/grades/index.json", "results/manifest.json"]} by="the sampling runs, goodhart grade and goodhart manifest" />}
      >
        {d ? <DotRows rows={rows(d.methods, d.floor)} reference="M0" testId="methods-rows" /> : null}
      </Figure>
    </div>
  );
}

async function loadCoverage(set: string) {
  const index = await loadIndex();
  const byMethod = methodRuns(index);
  const requests = Object.values(byMethod).flat();
  const cov = await coverage(index, set, requests);
  const lines = METHODS.map((m) => {
    const runs = byMethod[m.code as MethodCode] ?? [];
    const row = [...cov.rows].reverse().find((r) => runs.includes(r.request));
    return { key: m.code, label: m.code, name: `${m.code} ${m.name}`, color: `var(--gh-${m.token})`, points: row?.points ?? [] };
  });
  return { sql: cov.sql, lines };
}

/** pass at k for k in the registered list, one line per method. */
export function Coverage() {
  const [set, setSet] = useState("test");
  const load = useLoad(() => loadCoverage(set), [set]);
  const d = load.value;
  const label = HELD_OUT.find((s) => s.key === set)?.label ?? set;
  const at = (l: { points: { k: number; value: number | null }[] }, k: number) => l.points.find((p) => p.k === k)?.value ?? null;
  const drawn = d?.lines.filter((l) => l.points.some((p) => p.value !== null)) ?? [];
  const base = drawn.find((l) => l.key === "M0");
  const message = d
    ? coverageMessage(base ? { label: base.name, at1: at(base, 1), at16: at(base, 16) } : null, drawn.filter((l) => l.key !== "M0").map((l) => ({ label: l.name, at1: at(l, 1), at16: at(l, 16) })))
    : null;
  return (
    <div className="space-y-3">
      <SetPicker value={set} onChange={setSet} />
      <Figure
        testId="coverage-chart"
        message={message}
        subtitle={`The unbiased pass at k from the samples drawn per question at the registered temperature, twin confirmed, on the ${label} set. A method that only sharpens its first answer gains at the smallest k and little at the largest; one that widens what it can reach gains at both.`}
        sources={["results/grades/index.json"]}
        sql={d?.sql || null}
        sqlNote="The query counts samples and twin confirmed samples per item; the page applies the unbiased estimator, one minus C(n minus c, k) over C(n, k), to each item and averages over items."
        table={{
          columns: ["method", ...KS.map((k) => `pass at ${k}`)],
          formats: ["text", ...KS.map(() => "pct1")],
          rows: (d?.lines ?? []).map((l) => [l.name, ...KS.map((k) => at(l, k))]),
        }}
        state={stateOf(load, drawn.length > 0)}
        error={load.error?.message}
        skeleton="lines"
        missing={<NotYetRun what={`pass at k for every method on the ${label} set.`} reads={["results/grades/index.json"]} by="the sampling runs and goodhart grade" />}
      >
        <CoverageChart lines={drawn} ks={[...KS]} testId="coverage-lines" />
      </Figure>
    </div>
  );
}

/** The regression suite and the public benchmark, from the manifest table methods.regression. */
export function Regression() {
  const load = useLoad(loadManifest, []);
  const rowsR = load.value?.rows("methods.regression") ?? [];
  return (
    <Figure
      testId="regression"
      message={
        rowsR.length
          ? `IFEval and Spider dev, graded for ${formatValue(new Set(rowsR.map((r) => String(r.method))).size, "int")} checkpoints: what training on three warehouses did to what the base could already do.`
          : "The regression suite has not been graded yet."
      }
      subtitle="IFEval in full by its reference checker, the change against the base with its interval; Spider dev under the test suite evaluation, greedy, for the base and the champion, labelled as a public benchmark likely present in every candidate's pretraining data."
      sources={["results/manifest.json"]}
      state={stateOf(load, rowsR.length > 0)}
      error={load.error?.message}
      skeleton="table"
      height={120}
      table={{
        columns: ["method", "suite", "score", "low", "high"],
        formats: ["text", "text", "pct1", "pct1", "pct1"],
        rows: rowsR.map((r) => [r.method ?? null, r.suite ?? null, r.point ?? null, r.low ?? null, r.high ?? null]),
      }}
      missing={<NotYetRun what="IFEval for every method with its change against the base, and Spider dev for the base and the champion." reads={["results/manifest.json"]} by="the sampling runs and goodhart manifest (table methods.regression)" />}
    >
      <DataTable
        table={{
          columns: ["method", "suite", "score", "low", "high"],
          formats: ["text", "text", "pct1", "pct1", "pct1"],
          rows: rowsR.map((r) => [r.method ?? null, r.suite ?? null, r.point ?? null, r.low ?? null, r.high ?? null]),
        }}
      />
    </Figure>
  );
}

/** The registered hypotheses, each with its verdict whatever it is. */
export function HypothesesList() {
  const hyp = useSource<Hypotheses>("evaluation/hypotheses.yaml");
  const man = useLoad(loadManifest, []);
  if (hyp.status === "loading") return <div className="skeleton h-40 sheet" aria-busy="true" />;
  if (!hyp.value) return <NotYetRun what="The registered hypotheses." reads={["evaluation/hypotheses.yaml"]} />;
  const verdicts = man.value?.rows("hypotheses.verdicts") ?? [];
  return (
    <ol className="space-y-3" data-testid="hypotheses">
      {Object.entries(hyp.value.hypotheses).map(([id, h]) => {
        const v = verdicts.find((r) => String(r.id) === id);
        const held = v ? (typeof v.held === "boolean" ? v.held : null) : null;
        const word = verdict(held);
        return (
          <li key={id} className="sheet p-4" data-hypothesis={id}>
            <p className="flex flex-wrap items-baseline justify-between gap-2">
              <span>
                <span className="font-mono text-ink2 mr-2">{id}</span>
                <span className="font-semibold">{h.statement}</span>
              </span>
              <span className={`text-sm font-semibold ${word === "held" ? "text-checks" : word === "did not hold" ? "text-reward" : "text-ink2"}`}>{word}</span>
            </p>
            <p className="text-sm text-ink2 mt-1">Test: {h.test}.</p>
            {v && typeof v.point === "number" ? (
              <p className="text-sm mt-1">
                Estimate {formatValue(v.point, "float3")}
                {typeof v.low === "number" && typeof v.high === "number" ? `, interval ${formatValue(v.low, "float3")} to ${formatValue(v.high, "float3")}` : ""}
                {typeof v.p_adjusted === "number" ? `, adjusted p ${formatValue(v.p_adjusted, "float3")}` : ""}.
              </p>
            ) : (
              <p className="text-sm mt-1 text-ink2">Not measured yet: the verdict comes from the repository (results/manifest.json, table hypotheses.verdicts).</p>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** The recommendation per use case, by rule from the readback test results. */
export function Recommendation() {
  const load = useLoad(() => loadMethods("readback_test"), []);
  const plan = useSource<Plan>("evaluation/plan.yaml");
  const d = load.value;
  if (load.status === "loading") return <div className="skeleton h-32 sheet" aria-busy="true" />;
  const scored = d?.methods.filter((m) => m.score?.twin_confirmed != null) ?? [];
  const ceiling = plan.value?.compute?.ceiling_gpu_hours_per_method ?? null;
  if (!d || scored.length < 2) {
    return (
      <div className="sheet p-4 measure" data-testid="recommendation">
        <p className="font-semibold">No recommendation yet.</p>
        <p className="text-sm mt-1">
          A recommendation is chosen by rule once at least the base and one trained method are graded on the readback test set; every method runs under the same
          ceiling{ceiling !== null ? ` of ${formatValue(ceiling, "hours1")} of GPU time` : ""}, counted from the run ledger.
        </p>
      </div>
    );
  }
  const best = [...scored].sort((a, b) => b.score!.twin_confirmed! - a.score!.twin_confirmed! || a.code.localeCompare(b.code))[0]!;
  const base = d.methods.find((m) => m.code === "M0");
  const perHour = scored.filter((m) => m.code !== "M0" && m.gpuHours && base?.score?.twin_confirmed != null).map((m) => ({ m, gain: (m.score!.twin_confirmed! - base!.score!.twin_confirmed!) / m.gpuHours! }));
  const cheapest = perHour.sort((a, b) => b.gain - a.gain)[0];
  const diff = best.code !== "M0" ? differenceFor(d.manifest, best.code, "M0", "readback_test", "twin_confirmed_accuracy") : null;
  return (
    <ul className="space-y-3" data-testid="recommendation">
      <li className="sheet p-4">
        <p className="font-semibold">With a verifier and the full budget</p>
        <p className="text-sm mt-1">
          {best.label} reached the highest twin confirmed accuracy on the readback test set, {formatValue(best.score!.twin_confirmed, "pct1")}.{" "}
          {best.code !== "M0" ? compare(best.label, "the base", diff) : "No trained method beat the base."}
        </p>
      </li>
      {cheapest ? (
        <li className="sheet p-4">
          <p className="font-semibold">When the GPU hour is the constraint</p>
          <p className="text-sm mt-1">
            {cheapest.m.label} bought the most accuracy per GPU hour over the base: {formatValue(cheapest.gain, "pts1")} per hour, from the run ledger.
          </p>
        </li>
      ) : null}
      <li className="sheet p-4">
        <p className="font-semibold">With no GPU at all</p>
        <p className="text-sm mt-1">
          The base with its fixed examples scored {formatValue(base?.score?.twin_confirmed ?? null, "pct1")}
          {d.floor.score ? `, and readback's no model floor ${formatValue(d.floor.score.twin_confirmed, "pct1")}` : ""}: the floor to beat before any training is worth paying for.
        </p>
      </li>
    </ul>
  );
}
