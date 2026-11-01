"use client";

import { NotYetRun } from "@/components/NotYetRun";
import { Script } from "@/components/Script";
import { loadUrl } from "@/lib/data";
import { queryGrades } from "@/lib/duck";
import { flaggedSql, gradeFiles, type MarkRow } from "@/lib/grades";
import { FLAG_OF_STAT, loadHacks, type Outcome } from "@/lib/hacks";
import { flagsOf, marksFromRow } from "@/lib/marks";
import type { ScriptSet } from "@/lib/types";
import { useLoad } from "@/lib/useLoad";

interface ArmScript {
  outcome: Outcome;
  sql: string;
  scripts: { row: MarkRow & { flagged: boolean }; question: string; text: string | null }[];
}

async function loadArmScripts(): Promise<ArmScript[]> {
  const d = await loadHacks();
  const dev = d.index.scripts.dev ? await loadUrl<ScriptSet>(d.index.scripts.dev.url) : null;
  const out: ArmScript[] = [];
  for (const o of d.outcomes.filter((x) => x.kind !== "honest" && x.run && x.lastCheckpoint)) {
    const flag = FLAG_OF_STAT[o.prediction?.detector ?? ""] ?? "flag_valid_but_wrong";
    const sql = flaggedSql(o.run!, o.lastCheckpoint!, flag);
    const files = gradeFiles(d.index, [o.run!], ["dev"]);
    const rows = files.length ? await queryGrades<MarkRow & { flagged: boolean }>(sql, files) : [];
    out.push({
      outcome: o,
      sql,
      scripts: rows.map((row) => {
        const item = dev?.items[row.item_id];
        const answer = item?.answers.find((a) => a.run === o.run && a.checkpoint === row.checkpoint && a.sample === row.sample);
        return { row, question: item?.question ?? row.item_id, text: answer?.text ?? null };
      }),
    });
  }
  return out;
}

/** Three scripts from each arm's last checkpoint, the ones its predicted detector flagged first, with their marks. */
export function ArmScripts() {
  const load = useLoad(loadArmScripts, []);
  if (load.status === "loading") {
    return (
      <div className="space-y-3" aria-busy="true">
        {[0, 1].map((i) => (
          <div key={i} className="script h-40 skeleton" />
        ))}
      </div>
    );
  }
  if (load.status === "error") return <p role="alert">The scripts did not load: {load.error.message}.</p>;
  if (!load.value.length) {
    return <NotYetRun what="Three answers from each arm's last checkpoint, with what its reward paid and what the twins confirmed." reads={["results/grades/index.json"]} by="the defect study runs, graded by goodhart grade" />;
  }
  return (
    <div className="space-y-8">
      {load.value.map(({ outcome, scripts }) => (
        <div key={outcome.arm} data-testid="arm-scripts" data-arm={outcome.arm}>
          <h3 className="text-[1.15rem]">{outcome.label}</h3>
          <p className="text-sm text-ink2 mb-3">
            {outcome.lastCheckpoint}, dev, greedy. {scripts.some((s) => s.row.flagged) ? "Flagged by the predicted detector first." : "The predicted detector flagged none of these answers."}
          </p>
          <div className="space-y-3">
            {scripts.map(({ row, question, text }) => (
              <Script
                key={`${row.item_id}-${row.sample}`}
                question={question}
                meta={`${row.item_id} · ${outcome.run} · ${row.checkpoint}`}
                who={`${outcome.label}, ${row.checkpoint}`}
                answer={text ?? "(the completion text is not in the published scripts)"}
                marks={marksFromRow(row, outcome.arm)}
                flags={flagsOf(row)}
                note={outcome.arm === "lenient" || outcome.arm === "format" || outcome.arm === "random" ? `What the ${outcome.arm} reward paid for this answer is not in the grade, so only the honest reward is marked.` : undefined}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
