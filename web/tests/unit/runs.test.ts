import { describe, expect, it } from "vitest";

import type { DataIndex, RunEntry } from "@/lib/data";
import { armOfRun, armRuns, methodOfRun, methodRuns, paidOf, stepOf } from "@/lib/runs";

const run = (fields: Partial<RunEntry>): RunEntry => ({ files: {}, completions: [], ...fields });

describe("which run is which", () => {
  it("reads a method from the request's method, then from an arm spelled as a method", () => {
    expect(methodOfRun(run({ method: "M3" }))).toBe("M3");
    expect(methodOfRun(run({ arm: "sft" }))).toBe("M1");
    expect(methodOfRun(run({ arm: "SFT then GRPO" }))).toBe("M4");
    expect(methodOfRun(run({ arm: "lenient" }))).toBeNull();
  });

  it("reads a defect study arm only from a GRPO run that names no method", () => {
    expect(armOfRun(run({ stage: "grpo", arm: "lenient" }))).toBe("lenient");
    expect(armOfRun(run({ stage: "grpo", arm: null, reward: "honest" }))).toBe("honest");
    expect(armOfRun(run({ stage: "grpo", arm: "honest", method: "M3" }))).toBeNull();
    expect(armOfRun(run({ stage: "sample", arm: "lenient" }))).toBeNull();
    expect(armOfRun(run({ stage: "grpo", arm: "kl" }))).toBeNull();
  });

  it("takes the latest run by id when an arm ran twice, and groups method runs", () => {
    const index = {
      runs: {
        "2026-11-01-01-grpo-lenient": run({ stage: "grpo", arm: "lenient" }),
        "2026-11-02-01-grpo-lenient": run({ stage: "grpo", arm: "lenient" }),
        "2026-11-01-02-sample-m0": run({ stage: "sample", method: "M0" }),
        "2026-11-01-03-sample-m0": run({ stage: "sample", method: "M0" }),
      },
    } as unknown as DataIndex;
    expect(armRuns(index)).toEqual({ lenient: "2026-11-02-01-grpo-lenient" });
    expect(methodRuns(index)).toEqual({ M0: ["2026-11-01-02-sample-m0", "2026-11-01-03-sample-m0"] });
  });

  it("finds the reward paid under any key the monitor or the trainer used, and the step of a checkpoint", () => {
    expect(paidOf({ reward: 0.4 })).toBe(0.4);
    expect(paidOf({ mean_reward: 0.5 })).toBe(0.5);
    expect(paidOf({ entropy: 0.2 })).toBeNull();
    expect(stepOf("step-125")).toBe(125);
    expect(stepOf("final")).toBeNull();
  });
});
