import { describe, expect, it } from "vitest";

import { compare, countWord, listWords, MIN_CLUSTERS, plural, sentence, verdict } from "@/lib/messages";

describe("compare, one branch per outcome of the interval", () => {
  it("says what is missing when one side was not measured", () => {
    expect(compare("M4", "M0", null)).toBe("M4 and M0 cannot be compared on twin confirmed accuracy: one of them was not measured.");
  });

  it("refuses to compare on too few clusters", () => {
    const text = compare("M4", "M0", { point: 0.2, low: 0.1, high: 0.3, clusters: MIN_CLUSTERS - 1 });
    expect(text).toBe(`M4 and M0 are not compared on twin confirmed accuracy: only ${MIN_CLUSTERS - 1} question clusters are behind the difference, fewer than the ${MIN_CLUSTERS} a comparison needs.`);
  });

  it("says beat when the whole interval is above zero", () => {
    expect(compare("M4", "M0", { point: 0.123, low: 0.05, high: 0.2, clusters: 12 })).toBe(
      "M4 beat M0 on twin confirmed accuracy by 12.3 points (95 percent interval +5.0 to +20.0 points).",
    );
  });

  it("says trailed when the whole interval is below zero", () => {
    expect(compare("M2", "M3", { point: -0.08, low: -0.15, high: -0.01, clusters: 12 }, "pass at 1")).toBe(
      "M2 trailed M3 on pass at 1 by 8.0 points (95 percent interval -15.0 to -1.0 points).",
    );
  });

  it("says within noise when the interval holds zero", () => {
    expect(compare("M1", "M0", { point: 0.01, low: -0.04, high: 0.06, clusters: 12 })).toBe(
      "M1 and M0 are within noise on twin confirmed accuracy: the difference is +1.0 points (95 percent interval -4.0 to +6.0 points), and the interval holds zero.",
    );
  });
});

describe("the small words", () => {
  it("prints the verdict whatever the outcome", () => {
    expect(verdict(true)).toBe("held");
    expect(verdict(false)).toBe("did not hold");
    expect(verdict(null)).toBe("not measured");
    expect(verdict(undefined)).toBe("not measured");
  });

  it("lists, counts and capitalizes", () => {
    expect(listWords([])).toBe("");
    expect(listWords(["a"])).toBe("a");
    expect(listWords(["a", "b"])).toBe("a and b");
    expect(listWords(["a", "b", "c"])).toBe("a, b and c");
    expect(countWord(0)).toBe("no");
    expect(countWord(4)).toBe("four");
    expect(countWord(16)).toBe("16");
    expect(countWord(1234)).toBe("1,234");
    expect(plural(1, "run")).toBe("one run");
    expect(plural(3, "run")).toBe("three runs");
    expect(sentence("two runs")).toBe("Two runs");
    expect(sentence("")).toBe("");
  });
});
