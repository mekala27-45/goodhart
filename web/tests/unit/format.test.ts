import { describe, expect, it } from "vitest";

import { FORMATS, formatValue } from "@/lib/format";

// Expected strings are what packages/core/src/goodhart_core/formats.py prints for the same values.
describe("formats, as the repository prints them", () => {
  it.each([
    [2482, "int", "2,482"],
    [0.045124899274778404, "pct1", "4.5%"],
    [0.3920225624496374, "pct1", "39.2%"],
    [0.1, "pts1", "+10.0 pts"],
    [-0.0004, "pts1", "0.0 pts"],
    [-0.25, "pts1", "-25.0 pts"],
    [0.25, "apts1", "25.0 pts"],
    [-12.5, "usd2", "-$12.50"],
    [-0.001, "usd2", "$0.00"],
    [1234.5, "usd0", "$1,234"],
    [1.825, "hours2", "1.82 h"],
    [0.125, "float2", "0.12"],
    [0.375, "float2", "0.38"],
    [70386241, "int", "70,386,241"],
    [11_200_000_000, "gb1", "11.2 GB"],
    [true, "bool", "yes"],
    ["M4", "text", "M4"],
  ])("%s as %s is %s", (value, fmt, expected) => {
    expect(formatValue(value, fmt)).toBe(expected);
  });

  it("prints a missing value as not available in every format", () => {
    for (const fmt of Object.keys(FORMATS)) expect(formatValue(null, fmt)).toBe("not available");
    expect(formatValue(Number.NaN, "pct1")).toBe("not available");
  });

  it("refuses a format it does not know", () => {
    expect(() => formatValue(1, "pct9")).toThrow("unknown format");
  });
});
