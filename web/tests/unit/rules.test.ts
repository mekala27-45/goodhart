import { describe, expect, it } from "vitest";

import {
  contaminationMessage,
  coverageMessage,
  defectsMessage,
  distillMessage,
  dropsMessage,
  execTwinMessage,
  heroMessage,
  ledgerMessage,
  methodsMessage,
  mutationMessage,
  panelLine,
  type PanelSummary,
  quotaMessage,
  splitsMessage,
} from "@/lib/rules";
import type { Contamination, MutationRow, TwinsLock } from "@/lib/types";

const panel = (p: Partial<PanelSummary>): PanelSummary => ({
  arm: "honest",
  label: "honest",
  ran: true,
  lastStep: 200,
  paid: 0.7,
  confirmed: 0.5,
  firedAt: null,
  window: null,
  ...p,
});

describe("the hero's title", () => {
  it("says nothing has run when no arm has", () => {
    expect(heroMessage([panel({ ran: false, paid: null, confirmed: null }), panel({ arm: "lenient", label: "lenient (D1)", ran: false, paid: null, confirmed: null })])).toBe(
      "No arm of the defect study has run yet: each panel shows only the window its prediction registered.",
    );
  });

  it("says runs without a graded checkpoint are not compared", () => {
    expect(heroMessage([panel({ paid: null, confirmed: null }), panel({ arm: "lenient", ran: false, paid: null, confirmed: null })])).toBe(
      "The runs that exist have no graded checkpoint yet, so nothing is compared.",
    );
  });

  it("names the arm with the widest gap and sets it against the honest arm, naming what has not run", () => {
    const text = heroMessage([
      panel({ paid: 0.7, confirmed: 0.5 }),
      panel({ arm: "lenient", label: "lenient (D1)", lastStep: 75, paid: 0.91, confirmed: 0.21 }),
      panel({ arm: "lazy", label: "lazy (D3)", ran: false, paid: null, confirmed: null }),
    ]);
    expect(text).toBe("The lenient (D1) arm was paid 91.0% at step 75 while the twins confirmed 21.0% on dev: a gap of 70.0 pts, against 20.0 pts for the honest arm. lazy (D3) has not run.");
  });

  it("says so when the honest arm has the widest gap", () => {
    expect(heroMessage([panel({ paid: 0.9, confirmed: 0.2 }), panel({ arm: "random", label: "random (C1)", paid: 0.3, confirmed: 0.25 })])).toContain("The honest arm ends with the widest gap");
  });

  it("says no run was overpaid when every gap is zero or below", () => {
    expect(heroMessage([panel({ paid: 0.4, confirmed: 0.5 })])).toBe("No run was paid more than the twins confirmed by its last checkpoint.");
  });

  it("writes one line per panel for every state", () => {
    expect(panelLine(panel({ ran: false, window: [25, 150] }))).toBe("Not yet run. Predicted to fire between steps 25 and 150.");
    expect(panelLine(panel({ ran: false }))).toBe("Not yet run.");
    expect(panelLine(panel({ paid: null }))).toBe("Running record without a graded checkpoint yet.");
    expect(panelLine(panel({ window: [25, 150], firedAt: 60 }))).toContain("the monitor fired at step 60, inside the predicted window");
    expect(panelLine(panel({ window: [25, 150], firedAt: 180 }))).toContain("outside the predicted window");
    expect(panelLine(panel({ window: [25, 150] }))).toContain("the monitor did not fire");
    expect(panelLine(panel({ firedAt: 12 }))).toContain("a monitor fired at step 12");
    expect(panelLine(panel({}))).toContain("no monitor fired");
  });
});

const all = (r: Partial<MutationRow>): MutationRow => ({ family: "all", mutants: 10, equivalent: 2, not_executed: 0, honest_paid_full: 2, lenient_paid: 4, twin_confirmed: 0, scored: 8, ...r });

describe("the verifier's titles", () => {
  it("compares the rewards on near misses", () => {
    expect(mutationMessage([all({})])).toBe("Of 8 near misses of the test golds, the lenient reward pays 50.0%, the honest reward pays 25.0% in full, and the twins confirm none of them.");
    expect(mutationMessage([all({ twin_confirmed: 1 })])).toContain("the twins confirm 12.5%");
    expect(mutationMessage([all({ lenient_paid: 2 })])).toContain("the lenient reward pays no more than the honest one");
    expect(mutationMessage([all({ scored: 0 })])).toBe("No mutant of a test gold was scored, so the rewards cannot be compared on near misses.");
    expect(mutationMessage([])).toContain("No mutant");
  });

  it("describes the greedy cover", () => {
    const lock = { kept: ["a", "b"], distillation: { gains: [5, 2], original_passing: 10, caught_by_twin: {}, items: 1, mutants: 1 } } as unknown as TwinsLock;
    expect(distillMessage(lock)).toBe("The two kept twins catch 7 of the 10 train and dev mutants the original warehouse pays; the first twin chosen catches 5 of them alone.");
    expect(distillMessage({ ...lock, distillation: { ...lock.distillation, original_passing: 0 } })).toContain("nothing to catch");
  });

  it("compares execution with twin confirmation", () => {
    expect(execTwinMessage([])).toBe("No checkpoint has been graded on a held-out set yet.");
    expect(execTwinMessage([{ label: "M0", execution: 0.5, twin: 0.5 }])).toContain("was also confirmed by the twins");
    expect(execTwinMessage([{ label: "M0", execution: 0.5, twin: 0.4 }, { label: "M4", execution: 0.7, twin: 0.5 }])).toBe(
      "Every checkpoint scores higher on the original than on the twins; the widest gap is M4, 20.0 pts.",
    );
    expect(execTwinMessage([{ label: "M0", execution: 0.5, twin: 0.5 }, { label: "M4", execution: 0.7, twin: 0.5 }])).toContain("Some checkpoint");
  });
});

describe("the data titles", () => {
  it("names the largest drop", () => {
    expect(dropsMessage({ "duplicate SQL": 5, "duplicate question": 2 })).toBe("7 candidates were dropped by two rules; the largest, duplicate SQL, took 5.");
    expect(dropsMessage({})).toBe("No candidate was dropped.");
  });

  it("sizes the splits", () => {
    expect(splitsMessage({ train: 100, test: 10, template_test: 5, readback_test: 3, dev: 9 })).toBe(
      "Training holds 100 questions; the three held-out sets hold 18 that share no entity tuple with it.",
    );
    expect(splitsMessage({})).toBe("There is no training set yet.");
  });

  it("reports contamination for each outcome", () => {
    const c = (p: Partial<Contamination>) => ({ exact: 0, tuple_overlap: 0, moved: [], ...p }) as Contamination;
    expect(contaminationMessage(c({}))).toContain("No held-out question matched training");
    expect(contaminationMessage(c({ moved: ["a", "b"] }))).toBe(
      "No held-out question matched a training question exactly or shared its entity tuple; 2 moved out of the test sets as near duplicates.",
    );
    expect(contaminationMessage(c({ exact: 1, moved: ["a"] }))).toBe("1 exact matches and 0 shared entity tuples were found; 1 items moved out of the test sets.");
  });

  it("counts the defects of the first look", () => {
    expect(defectsMessage([])).toBe("The first look found no defect to turn into a rule.");
    const d = (warehouse: string) => ({ id: warehouse, warehouse, found: "", count: 1, rule: "" });
    expect(defectsMessage([d("doorstep"), d("doorstep"), d("ledger")])).toBe("The first look turned three defects in the marts into generator rules; doorstep had the most, two.");
  });
});

describe("the methods titles", () => {
  const points = [
    { code: "M0", label: "M0 base", point: 0.3 },
    { code: "M4", label: "M4 SFT then GRPO", point: 0.6 },
  ];

  it("compares the champion with the base by the manifest's interval", () => {
    expect(methodsMessage(points, "M4", { point: 0.3, low: 0.1, high: 0.5, clusters: 9 }, "readback test")).toBe(
      "M4 SFT then GRPO beat M0 base on twin confirmed accuracy on readback test by 30.0 points (95 percent interval +10.0 to +50.0 points).",
    );
    expect(methodsMessage(points, "M4", null, "readback test")).toContain("cannot be compared");
  });

  it("names the best method when no champion is named, and says when nothing is graded", () => {
    expect(methodsMessage(points, null, null, "template test")).toContain("M4 SFT then GRPO has the highest twin confirmed accuracy on template test so far, 60.0%");
    expect(methodsMessage([{ code: "M0", label: "M0 base", point: null }], null, null, "template test")).toBe("No method has been graded on template test yet.");
  });

  it("reads pass at k as sharpening or widening", () => {
    expect(coverageMessage(null, [])).toContain("The base has no sampled answers yet");
    expect(coverageMessage({ label: "M0", at1: 0.2, at16: 0.5 }, [])).toBe("The base passes 20.0% at one sample and 50.0% at sixteen; no trained method has been sampled yet.");
    expect(coverageMessage({ label: "M0", at1: 0.2, at16: 0.5 }, [{ label: "M3", at1: 0.4, at16: 0.55 }])).toContain("sharpened the first answer");
    expect(coverageMessage({ label: "M0", at1: 0.2, at16: 0.5 }, [{ label: "M3", at1: 0.22, at16: 0.7 }])).toContain("the gain is in what sixteen tries reach");
  });
});

describe("the sizing titles", () => {
  it("sums the ledger against the ceilings", () => {
    expect(ledgerMessage([])).toBe("No run has written a ledger row yet.");
    expect(ledgerMessage([{ gpu_hours: 0.5, within_ceiling: true }, { gpu_hours: 0.25, within_ceiling: null }])).toBe(
      "Two runs used 0.75 h of GPU time, and every one stayed inside its ceiling.",
    );
    expect(ledgerMessage([{ gpu_hours: 2.5, within_ceiling: false }])).toBe("One run used 2.50 h of GPU time; 1 went past its ceiling.");
  });

  it("sets the quota plan against the weekly quota when the probe has read it", () => {
    expect(quotaMessage({ a: 1, mandatory_total: 18 }, 30)).toBe("The registered plan needs 18.0 h of a weekly quota of 30.0 h, 60.0% of it.");
    expect(quotaMessage({ a: 1, b: 2 }, null)).toContain("needs 3.0 h of GPU time before the optional seeds");
  });
});
