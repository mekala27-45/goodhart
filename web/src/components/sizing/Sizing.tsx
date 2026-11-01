"use client";

import { DataTable, Figure } from "@/components/charts/Figure";
import { HBars } from "@/components/charts/HBars";
import { linear, textWidth } from "@/components/charts/scale";
import { useWidth } from "@/components/charts/useWidth";
import { NotYetRun } from "@/components/NotYetRun";
import { loadIndex, loadRunLines } from "@/lib/data";
import { formatValue } from "@/lib/format";
import { quotaPlan } from "@/lib/jsonsql";
import { ledgerMessage, quotaMessage } from "@/lib/rules";
import type { GpuProbe, LedgerRow, Plan, Sizing } from "@/lib/types";
import { useLoad } from "@/lib/useLoad";
import { JSON_SQL_NOTE, stateOf, useSource } from "@/lib/useSource";

const PHASES = ["loading", "generation", "reward", "training", "evaluation", "other"] as const;
const PHASE_COLOR: Record<string, string> = {
  loading: "var(--gh-seq-1)",
  generation: "var(--gh-seq-2)",
  reward: "var(--gh-seq-3)",
  training: "var(--gh-seq-4)",
  evaluation: "var(--gh-seq-5)",
  other: "var(--gh-control)",
};

interface LedgerEntry extends LedgerRow {
  run: string;
  hours: number;
}

async function loadLedger(): Promise<{ rows: LedgerEntry[]; sources: string[] }> {
  const index = await loadIndex();
  const rows: LedgerEntry[] = [];
  const sources: string[] = [];
  for (const [id, entry] of Object.entries(index.runs).sort()) {
    if (!entry.files.ledger) continue;
    sources.push(entry.files.ledger);
    for (const r of await loadRunLines<LedgerRow>(entry, "ledger")) {
      const hours = Object.values(r.gpu_seconds ?? {}).reduce((s, v) => s + v, 0) / 3600;
      rows.push({ ...r, run: id, hours });
    }
  }
  return { rows, sources };
}

/** GPU time by phase for every run in the ledger, against its ceiling. */
export function LedgerChart() {
  const load = useLoad(loadLedger, []);
  const rows = load.value?.rows ?? [];
  return (
    <Figure
      testId="ledger-chart"
      message={load.value ? ledgerMessage(rows.map((r) => ({ gpu_hours: r.hours, within_ceiling: r.ceiling_hours ? r.hours <= r.ceiling_hours + 1e-9 : null }))) : null}
      subtitle="One row per stage run, measured on the GPU with a timer around each phase. The tick is the run's ceiling; a run that would pass it stops there and the ledger says so."
      sources={load.value?.sources.length ? load.value.sources : ["results/runs"]}
      table={{
        columns: ["run", "stage", "arm", "status", "steps", "prompts", "rollouts", "tokens generated", "tokens trained", "GPU hours", "ceiling", "stopped at ceiling", "peak memory"],
        formats: ["text", "text", "text", "text", "int", "int", "int", "int", "int", "hours3", "hours2", "bool", "gb1"],
        rows: rows.map((r) => [r.run, r.stage, r.arm ?? "", r.status, r.steps ?? null, r.prompts ?? null, r.rollouts ?? null, r.tokens_generated ?? null, r.tokens_trained ?? null, r.hours, r.ceiling_hours ?? null, Boolean(r.stopped_at_ceiling), r.peak_memory_bytes ?? null]),
      }}
      state={stateOf(load, rows.length > 0)}
      error={load.error?.message}
      skeleton="bars"
      missing={<NotYetRun what="The run ledger: prompts, rollouts, tokens generated and trained, GPU seconds by phase, peak memory and wall clock for every stage run." reads={["results/runs"]} by="each GPU run, promoted from its results branch" />}
    >
      <PhaseBars rows={rows} />
    </Figure>
  );
}

function PhaseBars({ rows }: { rows: LedgerEntry[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const labelH = 17;
  const barH = 14;
  const rowH = labelH + barH + 12;
  const height = rows.length * rowH + 24;
  const max = Math.max(1e-9, ...rows.map((r) => Math.max(r.hours, r.ceiling_hours ?? 0)));
  const x = linear([0, max], [0, Math.max(60, width - textWidth("0.00 h of 0.00 h", 12) - 12)]);
  return (
    <div ref={ref} className="w-full">
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink2 mb-2" aria-label="Phases">
        {PHASES.map((p) => (
          <li key={p} className="inline-flex items-center gap-1.5">
            <span className="inline-block w-3 h-3" style={{ background: PHASE_COLOR[p] }} aria-hidden="true" />
            {p}
          </li>
        ))}
      </ul>
      <svg width={width} height={height} role="img" aria-label="GPU hours by phase, one bar per run" className="block">
        {rows.map((r, i) => {
          const y = i * rowH;
          let at = 0;
          return (
            <g key={`${r.run}-${i}`} transform={`translate(0, ${y})`} data-run={r.run}>
              <text x={0} y={12} fontSize={12.5} className="fill-ink" fontWeight={600}>
                {r.stage}
                {r.arm ? `, ${r.arm}` : ""}
                <tspan className="fill-ink2 figure" fontWeight={400}>{`  ${r.run}`}</tspan>
              </text>
              {PHASES.map((p) => {
                const h = (r.gpu_seconds?.[p] ?? 0) / 3600;
                if (!h) return null;
                const x0 = x(at);
                at += h;
                return <rect key={p} x={x0} y={labelH} width={Math.max(0.8, x(at) - x0)} height={barH} fill={PHASE_COLOR[p]} stroke="var(--gh-raised)" strokeWidth={0.6} data-phase={p} />;
              })}
              {r.ceiling_hours ? <line x1={x(r.ceiling_hours)} x2={x(r.ceiling_hours)} y1={labelH - 3} y2={labelH + barH + 3} stroke="var(--gh-ink)" strokeWidth={2} /> : null}
              <text x={Math.max(x(r.hours), r.ceiling_hours ? x(r.ceiling_hours) : 0) + 6} y={labelH + barH - 3} fontSize={11.5} className="fill-ink2 figure">
                {formatValue(r.hours, "hours2")}
                {r.ceiling_hours ? ` of ${formatValue(r.ceiling_hours, "hours2")}` : ""}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** The registered quota plan, stage by stage, against the weekly quota the probe read. */
export function QuotaChart() {
  const planLoad = useSource<Plan>("evaluation/plan.yaml");
  const probeLoad = useSource<GpuProbe>("results/gpu_probe.json");
  const plan = planLoad.value;
  const probe = probeLoad.value;
  const rows = plan ? quotaPlan.rows(plan) : [];
  const quota = plan?.compute?.quota_plan ?? {};
  return (
    <Figure
      testId="quota-chart"
      message={plan ? quotaMessage(quota, probe?.weekly_quota_hours ?? null) : null}
      subtitle={
        probe
          ? `The probe ran on ${probe.accelerator} with ${probe.generation_backend} generation; the plan ${probe.fits_with_margin ? "fits" : "does not fit"} the weekly quota with margin.`
          : "The stages the registered plan budgets, in GPU hours. The probe reads the accelerator, the backend and steps per hour before the first run, and every run reads the probe."
      }
      sources={["evaluation/plan.yaml", "results/gpu_probe.json"]}
      sql={quotaPlan.sql}
      sqlNote={JSON_SQL_NOTE}
      table={{ columns: ["stage", "GPU hours"], formats: ["text", "hours1"], rows: rows.map((r) => [String(r.stage).replace(/_/g, " "), r.gpu_hours]) }}
      state={planLoad.status === "loading" || probeLoad.status === "loading" ? "loading" : stateOf(planLoad, rows.length > 0)}
      error={planLoad.error?.message}
      skeleton="bars"
      missing={<NotYetRun what="The quota plan of the registered evaluation plan." reads={["evaluation/plan.yaml"]} />}
    >
      <HBars
        format="hours1"
        rows={rows.map((r) => ({ label: String(r.stage).replace(/_/g, " "), values: { hours: r.gpu_hours as number } }))}
        series={[{ key: "hours", label: "budgeted", color: "var(--gh-seq-4)", texture: "solid" }]}
      />
    </Figure>
  );
}

/** The sizing sheet: measured runs and their projection to the sizes a customer would pay for, labelled as estimates. */
export function SizingSheet() {
  const load = useSource<Sizing>("results/sizing.json");
  const s = load.value;
  const runs = s?.runs ?? [];
  const projected = runs.filter((r) => r.projection && Object.keys(r.projection).length);
  const sizes = [...new Set(projected.flatMap((r) => Object.keys(r.projection ?? {})))].sort();
  return (
    <Figure
      testId="sizing-sheet"
      message={
        s
          ? `Every run measured here took ${formatValue(s.total_gpu_hours, "hours2")} of GPU time; the same recipes at ${sizes.join(" and ") || "larger sizes"} are projected below as estimates, never spent.`
          : "The sizing sheet has not been written yet."
      }
      subtitle={
        s
          ? `Projected from the measured token counts on the assumptions the sheet states: compute scales with parameters at the measured tokens, and an H100 does the measured work ${formatValue(Number(s.assumptions.speedup_t4_to_h100 ?? 0), "float1")} times faster. Prices read on ${String(s.assumptions.prices_read_on ?? "the day of the build")}.`
          : "Measured per run from the ledger, then projected to larger models from the token counts, priced from public price pages read on the day."
      }
      sources={["results/sizing.json"]}
      state={stateOf(load, Boolean(s))}
      error={load.error?.message}
      skeleton="table"
      table={{
        columns: ["run", "stage", "GPU hours", ...sizes.flatMap((z) => [`${z} H100 hours, estimate`, `${z} dollars, estimate`])],
        formats: ["text", "text", "hours3", ...sizes.flatMap(() => ["hours2", "usd2"])],
        rows: projected.map((r) => [r.request, r.stage, r.gpu_hours, ...sizes.flatMap((z) => [r.projection?.[z]?.h100_hours ?? null, r.projection?.[z]?.gpu_dollars ?? null])]),
      }}
      missing={<NotYetRun what="Per run throughput by phase and a projection of each recipe to larger models, priced from public price pages and labelled as an estimate." reads={["results/sizing.json"]} by="goodhart sizing" />}
    >
      <DataTable
        caption="Measured on the free GPU"
        table={{
          columns: ["run", "stage", "model", "prompts", "rollouts", "tokens generated", "tokens trained", "GPU hours", "generated tokens per second", "trained tokens per second", "peak memory"],
          formats: ["text", "text", "text", "int", "int", "int", "int", "hours3", "float1", "float1", "float1"],
          rows: runs.map((r) => [r.request, r.stage, r.model ?? "", r.prompts ?? null, r.rollouts ?? null, r.tokens_generated ?? null, r.tokens_trained ?? null, r.gpu_hours, r.generated_tokens_per_second ?? null, r.trained_tokens_per_second ?? null, r.peak_memory_gb ?? null]),
        }}
      />
      {projected.length ? (
        <div className="mt-4">
          <DataTable
            caption="Projected, an estimate, never spent"
            table={{
              columns: ["run", "stage", ...sizes.flatMap((z) => [`${z} H100 hours`, `${z} GPU dollars`, `${z} fine-tuning API dollars`])],
              formats: ["text", "text", ...sizes.flatMap(() => ["hours2", "usd2", "usd2"])],
              rows: projected.map((r) => [r.request, r.stage, ...sizes.flatMap((z) => [r.projection?.[z]?.h100_hours ?? null, r.projection?.[z]?.gpu_dollars ?? null, r.projection?.[z]?.fine_tuning_api_dollars ?? null])]),
            }}
          />
        </div>
      ) : null}
      {s?.assumptions.price_pages?.length ? (
        <p className="text-xs text-ink2 mt-3">
          Price pages:{" "}
          {s.assumptions.price_pages.map((u, i) => (
            <span key={u}>
              {i > 0 ? ", " : null}
              <a href={u}>{u}</a>
            </span>
          ))}
          .
        </p>
      ) : null}
    </Figure>
  );
}

/** The registered shrink order: what the plan gives up first when the probe says it does not fit. */
export function ShrinkOrder() {
  const load = useSource<Plan>("evaluation/plan.yaml");
  const text = load.value?.compute?.shrink_order;
  if (!text) return null;
  const [lead, ...steps] = String(text).split(";").map((s) => s.trim()).filter(Boolean);
  return (
    <div className="sheet p-4 sm:p-5 measure" data-testid="shrink-order">
      <p className="text-sm text-ink">{lead}.</p>
      <ol className="list-decimal pl-5 mt-2 text-sm space-y-1">
        {steps.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
    </div>
  );
}
