// The unbiased pass at k estimator, ported from packages/evaluation/src/goodhart_evaluation/metrics.py:
// 1 - C(n - c, k) / C(n, k), computed as a product so it stays exact for large n.
export function passAtK(n: number, c: number, k: number): number {
  if (k > n) throw new Error(`pass at ${k} needs at least ${k} samples, got ${n}`);
  if (c < 0 || c > n) throw new Error(`${c} correct of ${n} samples is impossible`);
  if (n - c < k) return 1;
  let product = 1;
  for (let i = n - c + 1; i <= n; i += 1) product *= 1 - k / i;
  return 1 - product;
}

/** The k values the plan registers. */
export const KS = [1, 2, 4, 8, 16] as const;

/** Mean pass at k over items, each item given as samples n and confirmed c; items with n < k are left out. */
export function meanPassAtK(items: { n: number; c: number }[], k: number): { value: number | null; items: number } {
  const usable = items.filter((i) => i.n >= k);
  if (!usable.length) return { value: null, items: 0 };
  const sum = usable.reduce((s, i) => s + passAtK(i.n, i.c, k), 0);
  return { value: sum / usable.length, items: usable.length };
}
