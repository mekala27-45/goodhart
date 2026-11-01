import { describe, expect, it } from "vitest";

import { judgeRows } from "@/lib/judge";

describe("the judge table", () => {
  it("reads each hack type's wrong answers, accepts and rate as goodhart judge writes them", () => {
    const rows = judgeRows({
      false_accept_by_hack: {
        literal_only: { wrong: 40, accepted: 12, rate: 0.3 },
        none: { wrong: 0, accepted: 0, rate: null },
      },
    });
    expect(rows).toEqual([
      ["literal_only", 40, 12, 0.3],
      ["none", 0, 0, null],
    ]);
  });

  it("has no rows before the judge has run", () => {
    expect(judgeRows(null)).toEqual([]);
  });
});
