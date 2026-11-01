"use client";

import { DotRows } from "@/components/charts/DotRows";
import { DataTable, Figure } from "@/components/charts/Figure";
import { NotYetRun } from "@/components/NotYetRun";
import { formatValue } from "@/lib/format";
import { judgeRows, type Judge } from "@/lib/judge";
import { loadMethods } from "@/lib/methods";
import { execTwinMessage } from "@/lib/rules";
import { useLoad } from "@/lib/useLoad";
import { stateOf, useSource } from "@/lib/useSource";

/** Execution accuracy on the original against twin confirmed accuracy, every method on the readback test set. */
export function ExecVsTwin() {
  const load = useLoad(() => loadMethods("readback_test"), []);
  const d = load.value;
  const rows = d ? [...d.methods, d.floor].filter((m) => m.score && m.score.execution !== null && m.score.twin_confirmed !== null) : [];
  return (
    <Figure
      testId="exec-twin-chart"
      message={d ? execTwinMessage(rows.map((m) => ({ label: m.label, execution: m.score!.execution!, twin: m.score!.twin_confirmed! }))) : null}
      subtitle="The hollow diamond is the share right on the original warehouse, what the honest reward checks; the dot is the share the twins also confirm. The distance between them is what the reward would have paid for answers that were right for the wrong reason."
      sources={["results/grades/index.json"]}
      sql={d?.sql}
      table={{
        columns: ["method", "execution accuracy", "twin confirmed accuracy", "answerable items"],
        formats: ["text", "pct1", "pct1", "int"],
        rows: rows.map((m) => [m.label, m.score!.execution, m.score!.twin_confirmed, m.score!.answerable]),
      }}
      state={stateOf(load, rows.length > 0)}
      error={load.error?.message}
      skeleton="dots"
      height={200}
      missing={<NotYetRun what="Execution accuracy against twin confirmed accuracy for every graded checkpoint." reads={["results/grades/index.json"]} by="goodhart grade" />}
    >
      <DotRows
        rows={rows.map((m) => ({ key: m.code, label: m.label, color: m.code === "floor" ? "var(--gh-control)" : `var(--gh-${m.token})`, point: m.score!.twin_confirmed, second: m.score!.execution, floor: m.code === "floor" }))}
        secondLabel="right on the original alone"
      />
    </Figure>
  );
}

/** The LLM judge against the verifier on the registered calibration set. */
export function JudgeCalibration() {
  const load = useSource<Judge>("results/judge.json");
  const j = load.value;
  const rows = judgeRows(j);
  const columns = ["hack type", "wrong answers", "accepted", "false accept rate"];
  const formats = ["text", "int", "int", "pct1"];
  return (
    <Figure
      testId="judge"
      message={
        j
          ? `The judge agreed with the verifier on ${formatValue(j.agreement ?? null, "pct1")} of the calibration set, a Cohen kappa of ${formatValue(j.kappa ?? null, "float2")}; as a reward it would have paid ${formatValue(j.would_have_paid ?? null, "pct1")} where the honest reward paid ${formatValue(j.honest_paid ?? null, "pct1")}, and it never enters a reward.`
          : "The judge has not been calibrated yet."
      }
      subtitle="A judge from a family different from every policy reads a registered, stratified set of completions whose truth the verifier already knows. The false accept rate per hack type says which hacks it would have paid for."
      sources={["results/judge.json"]}
      state={stateOf(load, Boolean(j))}
      error={load.error?.message}
      skeleton="table"
      height={120}
      table={{ columns, formats, rows }}
      missing={<NotYetRun what="Agreement, Cohen kappa, the false accept rate on each hack type, the padding test and the order test." reads={["results/judge.json"]} by="the judge kernel and goodhart grade" />}
    >
      {j ? (
        <>
          <DataTable caption="False accepts by hack type" table={{ columns, formats, rows }} />
          <p className="text-sm text-ink2 mt-2">
            The padding test appended a long comment to {formatValue(j.padding?.pairs ?? null, "int")} candidates and the verdict
            changed on {formatValue(j.padding?.rate ?? null, "pct1")} of them; the order test asked {formatValue(j.order?.pairs ?? null, "int")} pairs in both
            orders and the same letter came back both times on {formatValue(j.order?.rate ?? null, "pct1")}.
          </p>
        </>
      ) : null}
    </Figure>
  );
}

interface ParityRecord {
  base: string;
  compared: number;
  mismatches: number;
  examples?: unknown[];
}

/** The API's verdicts against the repository's on every exported completion. */
export function Parity() {
  const load = useSource<Record<string, ParityRecord>>("results/parity.json");
  const p = load.value;
  const entries = p ? Object.entries(p).filter(([, v]) => v && typeof v.compared === "number") : [];
  const message = entries.length
    ? entries.every(([, v]) => v.mismatches === 0 && v.compared > 0)
      ? `The API and the repository gave the same verdict on every completion compared: ${entries.map(([k, v]) => `${formatValue(v.compared, "int")} against the ${k} API`).join(", ")}.`
      : `The API and the repository disagreed: ${entries.map(([k, v]) => `${formatValue(v.mismatches, "int")} of ${formatValue(v.compared, "int")} against the ${k} API`).join(", ")}.`
    : "Parity has not been checked yet.";
  return (
    <Figure
      testId="parity"
      message={message}
      subtitle="scripts/check_parity.py sends every exported completion of the readback test set through the grader API and compares each verdict with the repository's."
      sources={["results/parity.json"]}
      state={stateOf(load, entries.length > 0)}
      error={load.error?.message}
      skeleton="table"
      height={90}
      table={{ columns: ["against", "address", "compared", "mismatches"], formats: ["text", "text", "int", "int"], rows: entries.map(([k, v]) => [k, v.base, v.compared, v.mismatches]) }}
      missing={<NotYetRun what="The API verdict against the repository verdict on every exported completion." reads={["results/parity.json"]} by="scripts/check_parity.py" />}
    >
      <DataTable table={{ columns: ["against", "address", "compared", "mismatches"], formats: ["text", "text", "int", "int"], rows: entries.map(([k, v]) => [k, v.base, v.compared, v.mismatches]) }} />
    </Figure>
  );
}
