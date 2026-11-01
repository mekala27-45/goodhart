// The marks in the margin of a script: a word and a number each, red for what a reward paid, green
// for what the held-out checks confirmed. From a repository grade (a row of the grades parquet) or
// from a live grade (the API's response).
import type { Mark } from "@/components/Script";

import type { Grade } from "./api";
import { formatValue } from "./format";
import { honestPays, paysFor, type GradeFacts } from "./pays";

const two = (v: number) => formatValue(v, "float2");

export interface GradeRowLike extends GradeFacts {
  confirmed: boolean;
  [key: string]: unknown;
}

/** Marks for a graded completion: the reward its run trained on (when the grade can say), the honest reward, and the checks. */
export function marksFromRow(row: GradeRowLike, reward: string | null): Mark[] {
  const marks: Mark[] = [];
  if (reward && reward !== "honest") {
    const paid = paysFor(reward, row);
    if (paid !== null) marks.push({ kind: "reward", word: reward, value: two(paid), title: `what the ${reward} reward paid` });
  }
  marks.push({ kind: "reward", word: "honest", value: two(honestPays(row)), title: "what the honest reward pays" });
  if (!row.refuse_item) {
    marks.push({ kind: "checks", word: "original", value: row.original ? "1" : "0", title: "matches the gold on the original warehouse" });
    marks.push({ kind: "checks", word: "confirmed", value: row.confirmed ? "1" : "0", title: "matches on the original and on every kept twin" });
  } else {
    marks.push({ kind: "checks", word: "refused", value: row.kind === "refuse" ? "1" : "0", title: "a refusal item: the right answer is REFUSE" });
  }
  return marks;
}

/** Marks for a live or recorded grade from the API: every reward it reports, then the checks. */
export function marksFromGrade(g: Grade): Mark[] {
  const order = ["honest", "lenient", "lazy", "refusal"];
  const rewards = [...order.filter((k) => k in g.pays), ...Object.keys(g.pays).filter((k) => !order.includes(k)).sort()];
  return [
    ...rewards.map((k) => ({ kind: "reward" as const, word: k, value: two(g.pays[k]!), title: `what the ${k} reward would pay` })),
    { kind: "checks", word: "original", value: g.checks.original ? "1" : "0", title: "matches the gold on the original warehouse" },
    { kind: "checks", word: "twins", value: `${g.checks.twins_passed} of ${g.checks.twins_total}`, title: "kept twins on which it matches that twin's gold" },
    { kind: "checks", word: "confirmed", value: g.checks.confirmed ? "1" : "0", title: "matches on the original and every kept twin" },
  ];
}

/** The detector flags set on a grade row or a live grade, by name. */
export function flagsOf(source: Record<string, unknown>): string[] {
  return Object.entries(source)
    .filter(([k, v]) => v === true && (k.startsWith("flag_") || ["literal_only", "copied_value", "unfiltered", "refusal_answerable", "valid_but_wrong"].includes(k)))
    .map(([k]) => k.replace(/^flag_/, ""));
}
