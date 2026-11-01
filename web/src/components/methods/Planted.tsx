"use client";

import { DataTable, Figure } from "@/components/charts/Figure";
import { NotYetRun } from "@/components/NotYetRun";
import { formatValue } from "@/lib/format";
import { loadManifest } from "@/lib/methods";
import { useLoad } from "@/lib/useLoad";
import { stateOf } from "@/lib/useSource";

/** SFT trained on the planted targets against SFT on the cleaned ones, from the manifest table methods.planted. */
export function PlantedSft() {
  const load = useLoad(loadManifest, []);
  const rows = load.value?.rows("methods.planted") ?? [];
  const clean = rows.find((r) => String(r.set) === "clean");
  const planted = rows.find((r) => String(r.set) === "planted");
  const message =
    clean && planted && typeof clean.twin_confirmed === "number" && typeof planted.twin_confirmed === "number"
      ? `SFT on the cleaned targets scored ${formatValue(clean.twin_confirmed, "pct1")} twin confirmed on dev, against ${formatValue(planted.twin_confirmed, "pct1")} on the planted ones: a difference of ${formatValue(clean.twin_confirmed - planted.twin_confirmed, "pts1")}.`
      : "SFT on the planted and the cleaned targets has not been compared yet.";
  const table = {
    columns: ["targets", "twin confirmed on dev", "low", "high"],
    formats: ["text", "pct1", "pct1", "pct1"],
    rows: rows.map((r) => [r.set ?? null, r.twin_confirmed ?? null, r.low ?? null, r.high ?? null]),
  };
  return (
    <Figure
      testId="planted-sft"
      message={message}
      subtitle="The same SFT recipe on the targets with the planted mutations left in, and on the targets the audit cleaned: what a data defect the audit did not catch would have cost."
      sources={["results/manifest.json"]}
      state={stateOf(load, rows.length > 0)}
      error={load.error?.message}
      skeleton="table"
      height={90}
      table={table}
      missing={<NotYetRun what="Twin confirmed accuracy of SFT on the planted targets against SFT on the cleaned ones." reads={["results/manifest.json"]} by="the SFT runs, goodhart grade and goodhart manifest (table methods.planted)" />}
    >
      <DataTable table={table} />
    </Figure>
  );
}
