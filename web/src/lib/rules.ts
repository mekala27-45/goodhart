// The title of every chart, written by a rule from the rows the chart draws. A rule has a branch for
// every outcome it can meet (nothing run, some run, a tie, an empty set), and tests/unit/rules.test.ts
// takes each branch. Figures are printed through the repository's formats.
import { formatValue } from "./format";
import { compare, countWord, type Difference, listWords, plural, sentence } from "./messages";
import type { Contamination, MutationRow, ProfileDefect, TwinsLock } from "./types";

const pct = (v: number) => formatValue(v, "pct1");
const int = (v: number) => formatValue(v, "int");
const pts = (v: number) => formatValue(Math.abs(v), "apts1");

export interface PanelSummary {
  arm: string;
  label: string;
  ran: boolean;
  lastStep: number | null;
  paid: number | null;
  confirmed: number | null;
  firedAt: number | null;
  window: [number, number] | null;
}

/** The gap a panel ends on: paid minus confirmed at the last checkpoint with both. */
export function gapOf(p: PanelSummary): number | null {
  return p.paid !== null && p.confirmed !== null ? p.paid - p.confirmed : null;
}

export function heroMessage(panels: PanelSummary[]): string {
  const ran = panels.filter((p) => p.ran && gapOf(p) !== null);
  const waiting = panels.filter((p) => !p.ran).map((p) => p.label);
  if (!ran.length) {
    return waiting.length === panels.length
      ? "No arm of the defect study has run yet: each panel shows only the window its prediction registered."
      : "The runs that exist have no graded checkpoint yet, so nothing is compared.";
  }
  const widest = [...ran].sort((a, b) => gapOf(b)! - gapOf(a)! || a.arm.localeCompare(b.arm))[0]!;
  const honest = ran.find((p) => p.arm === "honest");
  const tail = waiting.length ? ` ${listWords(waiting)} ${waiting.length === 1 ? "has" : "have"} not run.` : "";
  const gap = gapOf(widest)!;
  if (gap <= 0) {
    return `No run was paid more than the twins confirmed by its last checkpoint.${tail}`;
  }
  if (widest.arm === "honest") {
    return `The honest arm ends with the widest gap of any run, ${pts(gap)} between what it was paid and what the twins confirmed.${tail}`;
  }
  const against = honest && gapOf(honest) !== null ? `, against ${pts(gapOf(honest)!)} for the honest arm` : "";
  return `The ${widest.label} arm was paid ${pct(widest.paid!)} at step ${int(widest.lastStep ?? 0)} while the twins confirmed ${pct(widest.confirmed!)} on dev: a gap of ${pts(gap)}${against}.${tail}`;
}

/** One line under a panel. */
export function panelLine(p: PanelSummary): string {
  if (!p.ran) return p.window ? `Not yet run. Predicted to fire between steps ${int(p.window[0])} and ${int(p.window[1])}.` : "Not yet run.";
  if (p.paid === null || p.confirmed === null) return "Running record without a graded checkpoint yet.";
  const fired =
    p.firedAt === null
      ? p.window
        ? "the monitor did not fire"
        : "no monitor fired"
      : p.window
        ? `the monitor fired at step ${int(p.firedAt)}, ${p.firedAt >= p.window[0] && p.firedAt <= p.window[1] ? "inside" : "outside"} the predicted window`
        : `a monitor fired at step ${int(p.firedAt)}`;
  return `Paid ${pct(p.paid)}, confirmed ${pct(p.confirmed)} at step ${int(p.lastStep ?? 0)}; ${fired}.`;
}

export function mutationMessage(rows: MutationRow[]): string {
  const all = rows.find((r) => r.family === "all");
  if (!all || !all.scored) return "No mutant of a test gold was scored, so the rewards cannot be compared on near misses.";
  const share = (n: number) => (n === 0 ? "none" : pct(n / all.scored));
  const twin = all.twin_confirmed === 0 ? "the twins confirm none of them" : `the twins confirm ${share(all.twin_confirmed)}`;
  if (all.lenient_paid <= all.honest_paid_full) {
    return `Of ${int(all.scored)} near misses of the test golds, the lenient reward pays no more than the honest one (${share(all.honest_paid_full)}), and ${twin}.`;
  }
  return `Of ${int(all.scored)} near misses of the test golds, the lenient reward pays ${share(all.lenient_paid)}, the honest reward pays ${share(all.honest_paid_full)} in full, and ${twin}.`;
}

export function distillMessage(lock: TwinsLock): string {
  const d = lock.distillation;
  const covered = d.gains.reduce((s, g) => s + g, 0);
  const k = lock.kept.length;
  if (!d.original_passing) return "No mutant passed on the original warehouse, so the twins had nothing to catch.";
  const first = d.gains[0] ?? 0;
  return `The ${countWord(k)} kept twins catch ${int(covered)} of the ${int(d.original_passing)} train and dev mutants the original warehouse pays; the first twin chosen catches ${int(first)} of them alone.`;
}

export function dropsMessage(drops: Record<string, number>): string {
  const entries = Object.entries(drops).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (!entries.length) return "No candidate was dropped.";
  const [reason, n] = entries[0]!;
  const total = entries.reduce((s, [, v]) => s + v, 0);
  return `${int(total)} candidates were dropped by ${plural(entries.length, "rule")}; the largest, ${reason}, took ${int(n)}.`;
}

export function splitsMessage(sets: Record<string, number>): string {
  const train = sets.train ?? 0;
  const held = ["readback_test", "test", "template_test"].filter((s) => sets[s] !== undefined);
  const heldTotal = held.reduce((s, k) => s + (sets[k] ?? 0), 0);
  if (!train) return "There is no training set yet.";
  return `Training holds ${int(train)} questions; the ${countWord(held.length)} held-out sets hold ${int(heldTotal)} that share no entity tuple with it.`;
}

export function contaminationMessage(c: Contamination): string {
  const moved = c.moved.length;
  const clean = c.exact === 0 && c.tuple_overlap === 0;
  if (clean && moved === 0) return "No held-out question matched training, shared an entity tuple or came near a training question.";
  if (clean) return `No held-out question matched a training question exactly or shared its entity tuple; ${int(moved)} moved out of the test sets as near duplicates.`;
  return `${int(c.exact)} exact matches and ${int(c.tuple_overlap)} shared entity tuples were found; ${int(moved)} items moved out of the test sets.`;
}

export function defectsMessage(defects: ProfileDefect[]): string {
  if (!defects.length) return "The first look found no defect to turn into a rule.";
  const by = new Map<string, number>();
  for (const d of defects) by.set(d.warehouse, (by.get(d.warehouse) ?? 0) + 1);
  const most = [...by.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]!;
  return `The first look turned ${plural(defects.length, "defect")} in the marts into generator rules; ${most[0]} had the most, ${countWord(most[1])}.`;
}

export interface MethodPoint {
  code: string;
  label: string;
  point: number | null;
}

/** The methods chart title: the champion against the base by the manifest's interval, else the point estimates only. */
export function methodsMessage(points: MethodPoint[], champion: string | null, diff: Difference | null, setLabel: string): string {
  const measured = points.filter((p) => p.point !== null);
  if (!measured.length) return `No method has been graded on ${setLabel} yet.`;
  const base = points.find((p) => p.code === "M0");
  const champ = champion ? points.find((p) => p.code === champion) : null;
  if (champ && base && champ.code !== base.code) {
    return compare(champ.label, base.label, diff, `twin confirmed accuracy on ${setLabel}`);
  }
  const best = [...measured].sort((a, b) => b.point! - a.point! || a.code.localeCompare(b.code))[0]!;
  return `${best.label} has the highest twin confirmed accuracy on ${setLabel} so far, ${pct(best.point!)}; no champion is named until the promotion gates run.`;
}

export interface CoverageSummary {
  label: string;
  at1: number | null;
  at16: number | null;
}

export function coverageMessage(base: CoverageSummary | null, others: CoverageSummary[]): string {
  if (!base || base.at1 === null || base.at16 === null) return "The base has no sampled answers yet, so no gain in pass at k can be measured.";
  const with16 = others.filter((o) => o.at1 !== null && o.at16 !== null);
  if (!with16.length) return `The base passes ${pct(base.at1)} at one sample and ${pct(base.at16)} at sixteen; no trained method has been sampled yet.`;
  const best = [...with16].sort((a, b) => b.at1! - a.at1! || a.label.localeCompare(b.label))[0]!;
  const g1 = best.at1! - base.at1;
  const g16 = best.at16! - base.at16;
  if (g1 > g16) return `${best.label} raised pass at 1 by ${formatValue(g1, "pts1")} and pass at 16 by ${formatValue(g16, "pts1")}: training sharpened the first answer more than it widened what sixteen tries can reach.`;
  return `${best.label} raised pass at 16 by ${formatValue(g16, "pts1")} and pass at 1 by ${formatValue(g1, "pts1")}: the gain is in what sixteen tries reach, not in the first answer.`;
}

export function ledgerMessage(runs: { gpu_hours: number; within_ceiling?: boolean | null }[]): string {
  if (!runs.length) return "No run has written a ledger row yet.";
  const hours = runs.reduce((s, r) => s + r.gpu_hours, 0);
  const inside = runs.filter((r) => r.within_ceiling !== false).length;
  if (inside === runs.length) return sentence(`${plural(runs.length, "run")} used ${formatValue(hours, "hours2")} of GPU time, and every one stayed inside its ceiling.`);
  const over = runs.length - inside;
  return sentence(`${plural(runs.length, "run")} used ${formatValue(hours, "hours2")} of GPU time; ${int(over)} went past ${over === 1 ? "its ceiling" : "their ceilings"}.`);
}

export function quotaMessage(quota: Record<string, number>, weekly: number | null): string {
  const total = quota.mandatory_total ?? Object.entries(quota).filter(([k]) => k !== "mandatory_total" && !k.includes("seeds_two")).reduce((s, [, v]) => s + v, 0);
  if (weekly === null) return `The registered plan needs ${formatValue(total, "hours1")} of GPU time before the optional seeds; the weekly quota is read by the probe, which has not run.`;
  const share = total / weekly;
  return `The registered plan needs ${formatValue(total, "hours1")} of a weekly quota of ${formatValue(weekly, "hours1")}, ${pct(share)} of it.`;
}

export function execTwinMessage(rows: { label: string; execution: number; twin: number }[]): string {
  if (!rows.length) return "No checkpoint has been graded on a held-out set yet.";
  const gaps = rows.map((r) => ({ ...r, gap: r.execution - r.twin }));
  const widest = [...gaps].sort((a, b) => b.gap - a.gap || a.label.localeCompare(b.label))[0]!;
  if (widest.gap <= 0) return "Every graded checkpoint that matched the gold on the original was also confirmed by the twins.";
  const all = gaps.every((g) => g.gap > 0);
  return `${all ? "Every" : "Some"} checkpoint scores higher on the original than on the twins; the widest gap is ${widest.label}, ${pts(widest.gap)}.`;
}
