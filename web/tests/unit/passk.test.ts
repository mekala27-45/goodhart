import { describe, expect, it } from "vitest";

import { meanPassAtK, passAtK } from "@/lib/passk";

function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 1; i <= k; i += 1) r = (r * (n - k + i)) / i;
  return r;
}

describe("the unbiased pass at k", () => {
  it("equals 1 - C(n - c, k) / C(n, k) on every small case", () => {
    for (let n = 1; n <= 16; n += 1)
      for (let c = 0; c <= n; c += 1)
        for (let k = 1; k <= n; k += 1) {
          expect(passAtK(n, c, k)).toBeCloseTo(1 - choose(n - c, k) / choose(n, k), 12);
        }
  });

  it("is one when fewer than k samples are wrong, and zero when none is right", () => {
    expect(passAtK(16, 15, 2)).toBe(1);
    expect(passAtK(16, 0, 16)).toBe(0);
  });

  it("refuses k above n and impossible counts", () => {
    expect(() => passAtK(8, 2, 16)).toThrow("needs at least 16 samples");
    expect(() => passAtK(8, 9, 1)).toThrow("impossible");
  });

  it("averages over items and leaves out items with too few samples", () => {
    expect(meanPassAtK([{ n: 16, c: 16 }, { n: 16, c: 0 }, { n: 1, c: 1 }], 2)).toEqual({ value: 0.5, items: 2 });
    expect(meanPassAtK([{ n: 1, c: 1 }], 16)).toEqual({ value: null, items: 0 });
  });
});
