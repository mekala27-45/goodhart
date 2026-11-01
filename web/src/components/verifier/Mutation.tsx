"use client";

import { Figure, DataTable } from "@/components/charts/Figure";
import { HBars } from "@/components/charts/HBars";
import { NotYetRun } from "@/components/NotYetRun";
import { formatValue } from "@/lib/format";
import { mutationByFamily } from "@/lib/jsonsql";
import { countWord } from "@/lib/messages";
import { mutationMessage } from "@/lib/rules";
import type { Mutation } from "@/lib/types";
import { JSON_SQL_NOTE, stateOf, useSource } from "@/lib/useSource";

/** The verifier chart: near misses of every test gold, paid by each reward and confirmed by the twins, per family. */
export function MutationScore() {
  const load = useSource<Mutation>("results/mutation.json");
  const m = load.value;
  const rows = m ? mutationByFamily.rows(m) : [];
  return (
    <Figure
      testId="mutation-chart"
      message={m ? mutationMessage(m.test) : null}
      subtitle={`Each mutant is one deliberate near miss of a test question's gold SQL, written by sqlglot. Mutants equivalent to the gold on the original and on ${m ? `all ${countWord(m.candidates.length)}` : "every"} candidate twins are set aside; the bars are shares of the rest. The honest reward runs on the original only, so a near miss that is right on the original by coincidence is paid in full; the twins are there to catch it.`}
      sources={["results/mutation.json"]}
      sql={mutationByFamily.sql}
      sqlNote={JSON_SQL_NOTE}
      table={{
        columns: ["family", "mutants", "equivalent", "scored", "honest pays in full", "lenient pays", "twins confirm"],
        formats: ["text", "int", "int", "int", "int", "int", "int"],
        rows: rows.map((r) => [r.family, r.mutants, r.equivalent, r.scored, r.honest_paid_full, r.lenient_paid, r.twin_confirmed]),
      }}
      state={stateOf(load, Boolean(m))}
      error={load.error?.message}
      skeleton="bars"
      height={380}
      missing={<NotYetRun what="The share of near misses each reward pays and the twins confirm, per family." reads={["results/mutation.json"]} by="goodhart mutants" />}
    >
      <HBars
        testId="mutation-bars"
        rows={rows.map((r) => ({
          label: r.family === "all" ? `every family, ${formatValue(r.scored as number, "int")} scored` : `${r.family}, ${formatValue(r.scored as number, "int")} scored`,
          total: r.family === "all",
          values: { lenient: r.lenient_pays as number | null, honest: r.honest_pays as number | null, twins: r.twins_confirm as number | null },
        }))}
        series={[
          { key: "lenient", label: "lenient pays", color: "var(--gh-reward)", texture: "hatch" },
          { key: "honest", label: "honest pays in full", color: "var(--gh-reward)", texture: "solid" },
          { key: "twins", label: "twins confirm", color: "var(--gh-checks)", texture: "solid" },
        ]}
        max={Math.max(...rows.map((r) => Number(r.lenient_pays ?? 0)), 0.05)}
      />
    </Figure>
  );
}

const OPERATOR_KEYS = ["mutants", "equivalent", "honest_paid_full", "lenient_paid", "twin_confirmed"] as const;

/** By operator: which kind of near miss each check lets through. */
export function MutationByOperator() {
  const load = useSource<Mutation>("results/mutation.json");
  const m = load.value;
  if (!m) return null;
  const rows = Object.entries(m.by_operator_test).sort((a, b) => (b[1].honest_paid_full ?? 0) - (a[1].honest_paid_full ?? 0) || a[0].localeCompare(b[0]));
  return (
    <div className="sheet p-4 sm:p-5">
      <h3 className="font-sans text-base font-bold">By mutation operator, test golds</h3>
      <p className="text-sm text-ink2 mt-1 mb-3 measure">{operatorMessage(rows)}</p>
      <DataTable
        caption="Mutants of the test golds by operator"
        table={{
          columns: ["operator", "mutants", "equivalent", "honest pays in full", "lenient pays", "twins confirm"],
          formats: ["text", "int", "int", "int", "int", "int"],
          rows: rows.map(([op, v]) => [op, ...OPERATOR_KEYS.map((k) => v[k] ?? 0)]),
        }}
      />
    </div>
  );
}

/** The operator the honest reward pays most often, by rule. */
function operatorMessage(rows: [string, Mutation["by_operator_test"][string]][]): string {
  const paid = rows.filter(([, v]) => (v.honest_paid_full ?? 0) > 0);
  if (!paid.length) return "The honest reward paid no mutant of any operator in full.";
  const [op, v] = paid[0]!;
  const caught = (v.honest_paid_full ?? 0) - (v.twin_confirmed ?? 0);
  return `"${op}" is the near miss the honest reward pays in full most often, ${formatValue(v.honest_paid_full ?? 0, "int")} of ${formatValue(v.mutants ?? 0, "int")}; the twins catch ${formatValue(caught, "int")} of those.`;
}

/** Near misses the honest reward paid in full and the twins did not confirm, and equivalents read by hand. */
export function MutationExamples() {
  const load = useSource<Mutation>("results/mutation.json");
  const m = load.value;
  if (!m) return null;
  const paid = m.paid_not_confirmed_examples.slice(0, 4);
  const equivalent = m.equivalent_examples.slice(0, 3);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="sheet p-4 sm:p-5 min-w-0">
        <h3 className="font-sans text-base font-bold">Paid in full, caught by the twins</h3>
        <p className="text-sm text-ink2 mt-1 mb-3">Each query below matched the gold on the original warehouse and failed on at least one twin.</p>
        <ul className="space-y-3">
          {paid.map((e, i) => (
            <li key={`${e.item}-${i}`}>
              <p className="text-xs text-ink2">
                <span className="font-mono">{e.item}</span>, {e.operator}
              </p>
              <pre className="sql mt-1">{e.sql}</pre>
            </li>
          ))}
        </ul>
      </div>
      <div className="sheet p-4 sm:p-5 min-w-0">
        <h3 className="font-sans text-base font-bold">Equivalent, set aside</h3>
        <p className="text-sm text-ink2 mt-1 mb-3">
          A mutant that returns the gold's result on the original and on every candidate twin is no near miss at all. A sample is read by hand; these are from it.
        </p>
        <ul className="space-y-3">
          {equivalent.map((e, i) => (
            <li key={`${e.item}-${i}`}>
              <p className="text-xs text-ink2">
                <span className="font-mono">{e.item}</span>, {e.operator}
              </p>
              <pre className="sql mt-1">{e.sql}</pre>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
