"use client";

import { Figure } from "@/components/charts/Figure";
import { HBars } from "@/components/charts/HBars";
import { NotYetRun } from "@/components/NotYetRun";
import { formatValue } from "@/lib/format";
import { distillation, twinAlone } from "@/lib/jsonsql";
import { countWord } from "@/lib/messages";
import { distillMessage } from "@/lib/rules";
import type { TwinsLock } from "@/lib/types";
import { JSON_SQL_NOTE, stateOf, useSource } from "@/lib/useSource";

const SOURCE = "data/twins.lock.json";

/** The greedy cover: the kept twins in the order they were chosen and what each added. */
export function Distillation() {
  const load = useSource<TwinsLock>(SOURCE);
  const lock = load.value;
  const rows = lock ? distillation.rows(lock) : [];
  const total = lock?.distillation.original_passing ?? 1;
  const covered = rows.at(-1)?.caught_so_far as number | undefined;
  return (
    <Figure
      testId="distillation-chart"
      message={lock ? distillMessage(lock) : null}
      subtitle={
        lock
          ? `Of ${countWord(lock.config.candidates ?? lock.kept.length)} seeded candidate twins, the greedy cover keeps the one that catches the most mutants the original warehouse pays, then the one that adds the most, until ${countWord(lock.kept.length)} are kept. A full bar is the whole pool of ${formatValue(total, "int")}${covered !== undefined ? `, of which ${formatValue(total - covered, "int")} no kept twin catches` : ""}. Mutants here come from the train and dev golds only; the test golds never chose a twin.`
          : undefined
      }
      sources={[SOURCE]}
      sql={distillation.sql}
      sqlNote={JSON_SQL_NOTE}
      table={{
        columns: ["order", "twin", "adds", "caught so far", "paid on the original"],
        formats: ["int", "text", "int", "int", "int"],
        rows: rows.map((r) => [r.rank, r.twin, r.caught, r.caught_so_far, r.original_passing]),
      }}
      state={stateOf(load, Boolean(lock))}
      error={load.error?.message}
      skeleton="bars"
      height={220}
      missing={<NotYetRun what="The twins kept by the greedy cover and what each catches." reads={[SOURCE]} by="goodhart twins" />}
    >
      <HBars
        format="int"
        max={total}
        rows={rows.map((r) => ({ label: `${formatValue(r.rank as number, "int")}. ${r.twin}`, values: { adds: r.caught as number, so_far: r.caught_so_far as number } }))}
        series={[
          { key: "adds", label: "adds", color: "var(--gh-checks)", texture: "hatch" },
          { key: "so_far", label: "caught so far", color: "var(--gh-checks)", texture: "solid" },
        ]}
      />
    </Figure>
  );
}

/** Each candidate twin alone, the kept ones named. */
export function TwinsAlone() {
  const load = useSource<TwinsLock>(SOURCE);
  const lock = load.value;
  const rows = lock ? twinAlone.rows(lock) : [];
  const kept = new Set(lock?.kept ?? []);
  const best = rows[0];
  return (
    <Figure
      testId="twins-alone-chart"
      message={
        lock && best
          ? `Alone, the best candidate twin catches ${formatValue(best.caught as number, "int")} of the ${formatValue(lock.distillation.original_passing, "int")} mutants the original pays; no single twin is enough, which is why ${countWord(lock.kept.length)} are kept.`
          : null
      }
      subtitle="Each candidate scales every numeric measure by a seeded factor drawn away from one, with a small jitter per row, and permutes a share of values within ranking columns so orders change. Keys, labels and years are never touched."
      sources={[SOURCE]}
      sql={twinAlone.sql}
      sqlNote={JSON_SQL_NOTE}
      table={{ columns: ["twin", "caught alone", "kept"], formats: ["text", "int", "text"], rows: rows.map((r) => [r.twin, r.caught, kept.has(String(r.twin)) ? "kept" : ""]) }}
      state={stateOf(load, Boolean(lock))}
      error={load.error?.message}
      skeleton="bars"
      height={420}
      missing={<NotYetRun what="What each candidate twin catches alone." reads={[SOURCE]} by="goodhart twins" />}
    >
      <HBars
        format="int"
        barHeight={11}
        rows={rows.map((r) => ({ label: `${r.twin}${kept.has(String(r.twin)) ? " (kept)" : ""}`, values: { alone: r.caught as number } }))}
        series={[{ key: "alone", label: "catches", color: "var(--gh-checks)", texture: "solid" }]}
      />
    </Figure>
  );
}

const CONFIG_WORDS: Record<string, [string, string]> = {
  candidates: ["candidate twins per warehouse", "int"],
  keep: ["twins kept", "int"],
  factor_low: ["lowest scale factor", "float2"],
  factor_high: ["highest scale factor", "float2"],
  jitter: ["jitter per row", "pct1"],
  permute_share: ["share permuted in ranking columns", "pct0"],
  distill_items: ["golds whose mutants chose the twins", "int"],
  mutants_per_operator: ["mutants per operator per gold", "int"],
  seed: ["seed", "text"],
};

/** The twin rules as registered in the lock file. */
export function TwinRules() {
  const load = useSource<TwinsLock>(SOURCE);
  const lock = load.value;
  if (!lock) return null;
  return (
    <dl className="sheet p-4 sm:p-5 grid grid-cols-[1fr_auto] gap-x-6 gap-y-1 text-sm max-w-xl" data-testid="twin-rules">
      {Object.entries(CONFIG_WORDS)
        .filter(([k]) => lock.config[k] !== undefined)
        .map(([k, [word, fmt]]) => (
          <div key={k} className="contents">
            <dt className="text-ink2">{word}</dt>
            <dd className="num figure text-ink">{formatValue(lock.config[k] as number, fmt)}</dd>
          </div>
        ))}
      <div className="contents">
        <dt className="text-ink2">kept, in order</dt>
        <dd className="figure text-ink text-right">{lock.kept.join(", ")}</dd>
      </div>
    </dl>
  );
}
