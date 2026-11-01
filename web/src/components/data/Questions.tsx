"use client";

import { Figure } from "@/components/charts/Figure";
import { HBars } from "@/components/charts/HBars";
import { NotYetRun } from "@/components/NotYetRun";
import { formatValue } from "@/lib/format";
import { contaminationChecks, drops, sets } from "@/lib/jsonsql";
import { ManifestView, type Manifest } from "@/lib/manifest";
import { contaminationMessage, defectsMessage, dropsMessage, splitsMessage } from "@/lib/rules";
import type { Audit, Contamination, Profile, QuestionsManifest } from "@/lib/types";
import { JSON_SQL_NOTE, stateOf, useSource } from "@/lib/useSource";

const QM = "data/questions/manifest.json";
const FAMILIES = ["lookup", "comparison", "trend", "top_k", "multi", "refusal"];
const SET_WORDS: Record<string, string> = {
  train: "training",
  dev: "development",
  test: "generated test",
  template_test: "template test",
  readback_test: "readback test",
  readback_dev: "readback development",
};

export function SetsChart() {
  const load = useSource<QuestionsManifest>(QM);
  const m = load.value;
  const rows = m ? sets.rows(m) : [];
  return (
    <Figure
      testId="sets-chart"
      message={m ? splitsMessage(m.sets) : null}
      subtitle="Every set the generator wrote. The bar is the number of questions; schema recall is the share of questions whose gold tables are all in the schema block the catalog search builds for the prompt."
      sources={[QM]}
      sql={sets.sql}
      sqlNote={JSON_SQL_NOTE}
      table={
        m
          ? {
              columns: ["set", "questions", "schema recall", ...FAMILIES],
              formats: ["text", "int", "pct1", ...FAMILIES.map(() => "int")],
              rows: rows.map((r) => [SET_WORDS[String(r.set)] ?? r.set, r.questions, r.schema_recall, ...FAMILIES.map((f) => m.families[String(r.set)]?.[f] ?? 0)]),
            }
          : null
      }
      state={stateOf(load, Boolean(m))}
      error={load.error?.message}
      skeleton="bars"
      missing={<NotYetRun what="The size of every set and its families." reads={[QM]} by="goodhart data" />}
    >
      <HBars
        format="int"
        rows={rows.map((r) => ({ label: `${SET_WORDS[String(r.set)] ?? r.set}, schema recall ${formatValue(r.schema_recall as number, "pct1")}`, values: { questions: r.questions as number } }))}
        series={[{ key: "questions", label: "questions", color: "var(--gh-answer)", texture: "solid" }]}
      />
    </Figure>
  );
}

export function DropsChart() {
  const load = useSource<QuestionsManifest>(QM);
  const m = load.value;
  const rows = m ? drops.rows(m) : [];
  return (
    <Figure
      testId="drops-chart"
      message={m ? dropsMessage(m.drops) : null}
      subtitle={
        m
          ? `The generator wrote ${formatValue(m.candidates, "int")} candidates from the catalog; ${formatValue(m.pool, "int")} survived into the pool the splits were cut from. A gold that returns nothing, runs past its time limit, changes when run again, or returns the same result on the original and every twin is dropped, and so is a duplicate by question or by SQL.`
          : undefined
      }
      sources={[QM]}
      sql={drops.sql}
      sqlNote={JSON_SQL_NOTE}
      table={{ columns: ["reason", "dropped"], formats: ["text", "int"], rows: rows.map((r) => [r.reason, r.dropped]) }}
      state={stateOf(load, Boolean(m))}
      error={load.error?.message}
      skeleton="bars"
      missing={<NotYetRun what="Every candidate dropped, by the rule that dropped it." reads={[QM]} by="goodhart data" />}
    >
      <HBars
        format="int"
        rows={rows.map((r) => ({ label: String(r.reason), values: { dropped: r.dropped as number } }))}
        series={[{ key: "dropped", label: "dropped", color: "var(--gh-control)", texture: "hatch" }]}
      />
    </Figure>
  );
}

export function ContaminationChart() {
  const load = useSource<Contamination>("results/contamination.json");
  const c = load.value;
  const rows = c ? contaminationChecks.rows(c) : [];
  const t = c?.thresholds ?? {};
  return (
    <Figure
      testId="contamination-chart"
      message={c ? contaminationMessage(c) : null}
      subtitle={
        c
          ? `Checked: ${Object.entries(c.checked)
              .map(([k, v]) => `${formatValue(v, "int")} ${SET_WORDS[k] ?? k}`)
              .join(", ")}. An ${formatValue(Number(t.ngram ?? 0), "int")}-gram overlap above ${formatValue(Number(t.ngram_share ?? 0), "pct0")}, or an embedding cosine above ${formatValue(Number(t.embedding_cosine ?? 0), "float2")} with ${String(t.embedding_model ?? "a local model")}, flags a pair; a flagged test item moved out before the plan was hashed.`
          : undefined
      }
      sources={["results/contamination.json"]}
      sql={contaminationChecks.sql}
      sqlNote={JSON_SQL_NOTE}
      table={{ columns: ["check", "found"], formats: ["text", "int"], rows: rows.map((r) => [String(r.check_name).replace(/_/g, " "), r.found]) }}
      state={stateOf(load, Boolean(c))}
      error={load.error?.message}
      skeleton="bars"
      missing={<NotYetRun what="Exact matches, entity tuple overlap, n-gram overlap and embedding near duplicates between the held-out sets and training." reads={["results/contamination.json"]} by="goodhart contamination" />}
    >
      <HBars
        format="int"
        rows={rows.map((r) => ({ label: String(r.check_name).replace(/_/g, " "), values: { found: r.found as number } }))}
        series={[{ key: "found", label: "found", color: "var(--gh-answer)", texture: "solid" }]}
        max={Math.max(1, ...rows.map((r) => Number(r.found)))}
      />
    </Figure>
  );
}

export function FirstLook() {
  const load = useSource<Profile>("results/profile.json");
  const p = load.value;
  return (
    <Figure
      testId="defects"
      message={p ? defectsMessage(p.defects) : null}
      subtitle={p ? `Before a question was generated, every mart was profiled at full size: ${formatValue(p.tables, "int")} tables, ${formatValue(p.rows, "int")} rows. Each defect below became a generator or quarantine rule with a test.` : undefined}
      sources={["results/profile.json"]}
      table={{ columns: ["defect", "warehouse", "rows", "rule"], formats: ["text", "text", "int", "text"], rows: (p?.defects ?? []).map((d) => [d.id, d.warehouse, d.count, d.rule]) }}
      state={stateOf(load, Boolean(p))}
      error={load.error?.message}
      skeleton="table"
      missing={<NotYetRun what="The first look at every mart and the defects it found." reads={["results/profile.json"]} by="goodhart profile" />}
    >
      <ol className="grid gap-3 sm:grid-cols-2" data-testid="defect-list">
        {(p?.defects ?? []).map((d) => (
          <li key={d.id} className="border border-hairline p-3 bg-surface">
            <p className="text-xs text-ink2 font-mono">
              {d.warehouse} · {d.id} · {formatValue(d.count, "int")} rows
            </p>
            <p className="mt-1 text-sm text-ink">{d.found}.</p>
            <p className="mt-1 text-sm text-ink2">
              <span className="font-semibold text-ink">Rule: </span>
              {d.rule}.
            </p>
          </li>
        ))}
      </ol>
    </Figure>
  );
}

export function Planted() {
  const qm = useSource<QuestionsManifest>(QM);
  const audit = useSource<Audit>("results/audit.json");
  const m = qm.value;
  const a = audit.value;
  const message = a
    ? a.precision !== null && a.recall !== null
      ? `The audit found ${formatValue(a.true_positives, "int")} of the ${formatValue(a.planted, "int")} planted targets, a recall of ${formatValue(a.recall, "pct1")} at a precision of ${formatValue(a.precision, "pct1")}.`
      : `The audit flagged ${formatValue(a.flagged, "int")} targets against ${formatValue(a.planted, "int")} planted.`
    : m
      ? `${formatValue(m.planted, "int")} supervised targets were corrupted on purpose, and the key was hashed before the audit ran; the audit has not run yet.`
      : null;
  const ops = a ? Object.keys({ ...a.planted_by_operator, ...a.missed_by_operator }).sort() : [];
  return (
    <Figure
      testId="planted"
      message={message}
      subtitle="Known mutations (a year shifted by one, a predicate dropped, an aggregate swapped) were planted in a share of the SFT targets and listed in a key whose hash was committed first. The audit re-executes every target on the original and the twins, flags the ones that fail, and only then reads the key."
      sources={["data/questions/manifest.json", "results/audit.json"]}
      state={qm.status === "loading" || audit.status === "loading" ? "loading" : a ? "ready" : m ? "missing" : stateOf(qm, false)}
      error={qm.error?.message ?? audit.error?.message}
      skeleton="bars"
      height={160}
      table={{
        columns: ["operator", "planted", "missed"],
        formats: ["text", "int", "int"],
        rows: ops.map((o) => [o, a?.planted_by_operator[o] ?? 0, a?.missed_by_operator[o] ?? 0]),
      }}
      missing={<NotYetRun what="The audit's precision and recall against the planted key, and the SFT difference between the planted and the cleaned set." reads={["results/audit.json"]} by="goodhart audit" />}
    >
      {a ? (
        <HBars
          format="int"
          rows={ops.map((o) => ({ label: o, values: { planted: a.planted_by_operator[o] ?? 0, missed: a.missed_by_operator[o] ?? 0 } }))}
          series={[
            { key: "planted", label: "planted", color: "var(--gh-reward)", texture: "hatch" },
            { key: "missed", label: "missed", color: "var(--gh-control)", texture: "solid" },
          ]}
        />
      ) : null}
    </Figure>
  );
}

export function Paraphrases() {
  const load = useSource<Manifest>("results/manifest.json");
  const view = new ManifestView(load.value ?? null);
  const rows = view.rows("data.paraphrases");
  return (
    <Figure
      testId="paraphrases"
      message={rows.length ? `${formatValue(view.number("data.paraphrases_kept"), "int")} paraphrases survived the round trip and are labelled model written wherever they appear.` : "The paraphrase round trip has not run yet."}
      subtitle="A teacher from a family no candidate shares rewrites a share of training questions into natural wording. A paraphrase is kept only if the teacher, given the paraphrase alone, writes SQL whose result matches the gold on the original and on every kept twin."
      sources={["results/manifest.json"]}
      state={stateOf(load, rows.length > 0)}
      error={load.error?.message}
      skeleton="table"
      height={80}
      missing={<NotYetRun what="How many paraphrases the teacher wrote and how many survived the round trip." reads={["results/manifest.json"]} by="the selection kernel, graded by goodhart grade and goodhart manifest (table data.paraphrases)" />}
    />
  );
}
