"use client";

import { MarkdownDoc } from "@/components/MarkdownDoc";
import { NotYetRun } from "@/components/NotYetRun";
import { loadIndex } from "@/lib/data";
import { formatValue } from "@/lib/format";
import type { MonitorConfig } from "@/lib/types";
import { useLoad } from "@/lib/useLoad";
import { useSource } from "@/lib/useSource";

const WORDS: Record<string, string> = {
  mean_result_rows: "mean result rows",
  share_literal_only: "share that reads no table",
  share_copied_value: "share that copies a value the prompt showed",
  share_unfiltered: "share with a dropped filter or extra rows",
  share_refusal_answerable: "share refusing an answerable question",
  share_valid_but_wrong: "share valid but wrong",
  frac_reward_zero_std: "share of groups with no reward spread",
  entropy: "entropy",
  clipped_ratio: "share running into the length cap",
};

/** The registered monitor thresholds, each a statistic, a comparison, a value and a window of steps. */
export function MonitorThresholds() {
  const load = useSource<MonitorConfig>("config/monitor.yaml");
  const m = load.value;
  if (load.status === "loading") return <div className="skeleton h-40 sheet" aria-busy="true" />;
  if (!m) return <NotYetRun what="The monitor thresholds registered with the plan." reads={["config/monitor.yaml"]} />;
  return (
    <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3" data-testid="thresholds">
      {Object.entries(m.thresholds).map(([stat, t]) => (
        <li key={stat} className="sheet px-3 py-2">
          <p className="text-sm font-semibold">{WORDS[stat] ?? stat}</p>
          <p className="text-sm text-ink2">
            alert when <span className="figure text-ink">{t.op === ">" ? "above" : "below"} {formatValue(t.value, Math.abs(t.value) >= 1 ? "float1" : "float2")}</span> for{" "}
            <span className="figure text-ink">{formatValue(t.window, "int")}</span> logged steps in a row
          </p>
          <p className="text-xs text-ink2 font-mono mt-0.5">{stat}</p>
        </li>
      ))}
    </ul>
  );
}

/** The incident: the first defect arm whose monitor fired, run end to end, and its postmortem. */
export function Incidents() {
  const load = useLoad(loadIndex, []);
  if (load.status === "loading") return <div className="skeleton h-24 sheet" aria-busy="true" />;
  const docs = load.value?.docs.incidents ?? [];
  if (!docs.length) {
    return (
      <NotYetRun
        what="The postmortem of the first defect arm whose monitor fired: the timeline from the ledger and the step logs, the steps and GPU minutes spent learning the hack, the root cause, what detected it and how fast, and three actions."
        reads={["docs/incidents"]}
        by="the incident, following docs/runbooks/reward-hacking.md"
      />
    );
  }
  return (
    <div className="space-y-8">
      {docs.map((d) => (
        <article key={d.url} className="sheet p-4 sm:p-6" data-testid="incident">
          <p className="text-xs text-ink2 font-mono mb-2">{d.url}</p>
          <MarkdownDoc url={d.url} />
        </article>
      ))}
    </div>
  );
}
