// Message rules: a sentence that compares numbers is chosen by a rule from the data, never typed by
// hand. compare() and verdict() are ported from packages/evaluation/src/goodhart_evaluation/
// messages.py so a page and the briefing say the same thing about the same interval; every chart's
// title is one of the rules in rules.ts, built from these. tests/unit/messages.test.ts covers every branch.

export const MIN_CLUSTERS = 5;

export interface Difference {
  point: number;
  low: number;
  high: number;
  clusters: number;
}

const pts = (x: number) => (Math.abs(x) * 100).toFixed(1);
const signedPts = (x: number) => {
  const text = (x * 100).toFixed(1);
  if (Number(text) === 0) return (0).toFixed(1);
  return x > 0 ? `+${text}` : text;
};

/** One sentence comparing a with b on a metric, chosen by the interval of a minus b. */
export function compare(a: string, b: string, diff: Difference | null, metric = "twin confirmed accuracy"): string {
  if (!diff) return `${a} and ${b} cannot be compared on ${metric}: one of them was not measured.`;
  if (diff.clusters < MIN_CLUSTERS) {
    return `${a} and ${b} are not compared on ${metric}: only ${diff.clusters} question clusters are behind the difference, fewer than the ${MIN_CLUSTERS} a comparison needs.`;
  }
  const span = `(95 percent interval ${signedPts(diff.low)} to ${signedPts(diff.high)} points)`;
  if (diff.low > 0) return `${a} beat ${b} on ${metric} by ${pts(diff.point)} points ${span}.`;
  if (diff.high < 0) return `${a} trailed ${b} on ${metric} by ${pts(diff.point)} points ${span}.`;
  return `${a} and ${b} are within noise on ${metric}: the difference is ${signedPts(diff.point)} points ${span}, and the interval holds zero.`;
}

/** The word printed beside every registered prediction and hypothesis, whatever the outcome. */
export function verdict(held: boolean | null | undefined): "held" | "did not hold" | "not measured" {
  if (held === null || held === undefined) return "not measured";
  return held ? "held" : "did not hold";
}

/** "a", "a and b", "a, b and c". */
export function listWords(words: string[]): string {
  if (words.length <= 1) return words[0] ?? "";
  return `${words.slice(0, -1).join(", ")} and ${words.at(-1)}`;
}

/** A count in words for the small counts a sentence needs; larger counts stay as figures. */
export function countWord(n: number): string {
  const words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
  return Number.isInteger(n) && n >= 0 && n < words.length ? words[n]! : n.toLocaleString("en-US");
}

/** A sentence that starts with a capital, whatever word a rule put first. */
export function sentence(text: string): string {
  return text ? text[0]!.toUpperCase() + text.slice(1) : text;
}

/** "1 run", "2 runs". */
export function plural(n: number, one: string, many = `${one}s`): string {
  return `${countWord(n)} ${n === 1 ? one : many}`;
}
