"use client";

import { DataTable, Figure } from "@/components/charts/Figure";
import { MarkdownDoc } from "@/components/MarkdownDoc";
import { NotYetRun } from "@/components/NotYetRun";
import { loadIndex, loadSource } from "@/lib/data";
import { formatValue, type Scalar } from "@/lib/format";
import { loadManifest } from "@/lib/methods";
import type { GpuProbe, Plan } from "@/lib/types";
import { useLoad } from "@/lib/useLoad";
import { stateOf, useSource } from "@/lib/useSource";

interface Selection {
  main?: string | null;
  small?: string | null;
  candidates?: Record<string, Scalar>[];
  [key: string]: unknown;
}

const PREFERRED = ["model", "license", "parameters", "passes", "greedy_pass_at_1", "pass_at_1", "pass_at_8", "tokens_per_second", "steps_per_hour_estimate", "why_not"];
const FORMATS: Record<string, string> = {
  parameters: "int",
  greedy_pass_at_1: "pct1",
  pass_at_1: "pct1",
  pass_at_8: "pct1",
  tokens_per_second: "float1",
  steps_per_hour_estimate: "float1",
};

async function loadSelection() {
  const [probe, selection] = await Promise.all([loadSource<GpuProbe>("results/gpu_probe.json"), loadSource<Selection>("results/selection.json")]);
  const byModel = new Map<string, Record<string, Scalar>>();
  for (const c of probe?.candidates ?? []) byModel.set(c.model, { ...(c as unknown as Record<string, Scalar>) });
  for (const c of selection?.candidates ?? []) {
    const model = String(c.model ?? "");
    if (model) byModel.set(model, { ...(byModel.get(model) ?? {}), ...c });
  }
  return { probe, selection, rows: [...byModel.values()] };
}

/** Every candidate through the probe and the dev sets, with its license, and the rule's choice. */
export function BaseSelection() {
  const load = useLoad(loadSelection, []);
  const plan = useSource<Plan>("evaluation/plan.yaml");
  const rule = (plan.value?.base_selection as { rule?: string } | undefined)?.rule ?? null;
  const d = load.value;
  const rows = d?.rows ?? [];
  const keys = PREFERRED.filter((k) => rows.some((r) => r[k] !== undefined && r[k] !== null));
  const main = d?.selection?.main ?? null;
  const small = d?.selection?.small ?? null;
  const passed = rows.filter((r) => r.passes === true).length;
  const message = !rows.length
    ? null
    : main || small
      ? `The rule chose ${main ?? "no main model"} as the main model and ${small ?? "no small model"} as the small one, from ${formatValue(rows.length, "int")} candidates of which ${formatValue(passed, "int")} passed the probe.`
      : `${formatValue(passed, "int")} of ${formatValue(rows.length, "int")} candidates passed the probe; the selection on the dev sets has not run yet.`;
  const table = {
    columns: keys.map((k) => k.replace(/_/g, " ")),
    formats: keys.map((k) => FORMATS[k] ?? "text"),
    rows: rows.map((r) => keys.map((k) => (typeof r[k] === "boolean" ? (r[k] ? "yes" : "no") : (r[k] ?? null)))),
  };
  return (
    <Figure
      testId="base-selection"
      message={message}
      subtitle={rule ? `The rule, registered with the plan before the probe ran: ${rule}.` : "The rule is registered with the plan before the probe runs."}
      sources={["results/gpu_probe.json", "results/selection.json", "evaluation/plan.yaml"]}
      state={stateOf(load, rows.length > 0)}
      error={load.error?.message}
      skeleton="table"
      table={table}
      missing={<NotYetRun what="Every candidate with its license, whether it passed the probe, its throughput, and its pass at 1 and pass at 8 on dev." reads={["results/gpu_probe.json", "results/selection.json"]} by="the probe and selection runs" />}
    >
      <DataTable table={table} dense={false} />
    </Figure>
  );
}

/** The registered promotion gates, each with the checkpoint that passed or failed it. */
export function PromotionGates() {
  const plan = useSource<Plan>("evaluation/plan.yaml");
  const man = useLoad(loadManifest, []);
  if (plan.status === "loading") return <div className="skeleton h-32 sheet" aria-busy="true" />;
  const gates = plan.value?.promotion_gates ?? [];
  const results = man.value?.rows("models.gates") ?? [];
  if (!gates.length) return <NotYetRun what="The promotion gates registered with the plan." reads={["evaluation/plan.yaml"]} />;
  return (
    <ol className="space-y-2" data-testid="gates">
      {gates.map((g, i) => {
        const mine = results.filter((r) => Number(r.gate) === i + 1 || String(r.gate) === g);
        return (
          <li key={g} className="sheet px-4 py-3">
            <p className="text-sm">
              <span className="font-mono text-ink2 mr-2">{formatValue(i + 1, "int")}</span>
              {g}
            </p>
            {mine.length ? (
              <ul className="mt-1 text-sm flex flex-wrap gap-x-4">
                {mine.map((r) => (
                  <li key={String(r.method)} className={r.passed ? "text-checks" : "text-reward"}>
                    {String(r.method)} {r.passed ? "passed" : "failed"}
                    {r.detail ? <span className="text-ink2">, {String(r.detail)}</span> : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-xs text-ink2">Not yet run.</p>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** Model cards or validation reports: every markdown file the repository holds in one folder, rendered. */
export function Documents({ kind }: { kind: "models" | "validation" }) {
  const load = useLoad(loadIndex, []);
  if (load.status === "loading") return <div className="skeleton h-24 sheet" aria-busy="true" />;
  const docs = load.value?.docs[kind] ?? [];
  if (!docs.length) {
    return kind === "models" ? (
      <NotYetRun
        what="One card per released checkpoint: the base and its license, the data and its digests, the method and every hyperparameter, compute from the ledger, results with intervals on every held-out set, the hack detector rates, known failure modes, intended use and the statement."
        reads={["docs/models"]}
        by="the claim gate renderer"
      />
    ) : (
      <NotYetRun
        what="A validation report for each champion candidate in SR 11-7 order: purpose, data, design, performance on development and test, stability across seeds, sensitivity from the interior tests, limitations, monitoring and the effective challenges on record."
        reads={["docs/validation"]}
        by="the claim gate renderer"
      />
    );
  }
  return (
    <div className="space-y-3">
      {docs.map((d, i) => (
        <details key={d.url} className="sheet" open={i === 0} data-testid={kind === "models" ? "model-card" : "validation-report"}>
          <summary className="cursor-pointer px-4 py-3 font-semibold">
            {d.name} <span className="text-xs text-ink2 font-mono font-normal">{d.url}</span>
          </summary>
          <div className="px-4 pb-5 sm:px-6">
            <MarkdownDoc url={d.url} />
          </div>
        </details>
      ))}
    </div>
  );
}
