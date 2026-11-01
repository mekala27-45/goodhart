"use client";

// Every defect arm and control: its registered prediction beside its outcome, with the word held or
// did not hold. A defect's outcome is computed here from the committed files, by the rule its
// prediction states: the predicted detector fired inside the predicted window (alerts.json), and by
// the arm's last checkpoint the twins confirmed at least the stated points less than the honest arm's
// last checkpoint on dev (the graded dev completions). A control's outcome needs the starting
// checkpoint's accuracy, so it is read from results/manifest.json (table hacks.outcomes) and is
// "not measured" until the repository writes it.
import { loadIndex, loadRunJson, loadSource, type DataIndex } from "./data";
import { setScores, type SetScore } from "./grades";
import { ManifestView, type Manifest } from "./manifest";
import { verdict } from "./messages";
import { ARM_LABEL, ARMS, type Arm, armRuns, CONTROLS, DEFECTS, stepOf } from "./runs";
import type { AlertRow, MonitorConfig, Prediction, Predictions } from "./types";

export interface Outcome {
  arm: Arm;
  label: string;
  kind: "honest" | "defect" | "control";
  run: string | null;
  prediction: Prediction | null;
  firedAt: number | null;
  firedStat: string | null;
  insideWindow: boolean | null;
  lastStep: number | null;
  lastCheckpoint: string | null;
  twin: number | null;
  overRefusal: number | null;
  gapPoints: number | null;
  overRefusalPoints: number | null;
  shareOfGain: number | null;
  held: boolean | null;
  word: ReturnType<typeof verdict>;
  reason: string;
}

/** The completion flag each monitored statistic is counted from. */
export const FLAG_OF_STAT: Record<string, string> = {
  mean_result_rows: "flag_unfiltered",
  share_literal_only: "flag_literal_only",
  share_copied_value: "flag_copied_value",
  share_unfiltered: "flag_unfiltered",
  share_refusal_answerable: "flag_refusal_answerable",
  share_valid_but_wrong: "flag_valid_but_wrong",
};

function lastRow(rows: SetScore[], run: string): SetScore | null {
  const mine = rows.filter((r) => r.request === run && stepOf(r.checkpoint) !== null);
  mine.sort((a, b) => (stepOf(a.checkpoint) ?? 0) - (stepOf(b.checkpoint) ?? 0));
  return mine.at(-1) ?? null;
}

export interface HacksData {
  index: DataIndex;
  predictions: Predictions | null;
  monitor: MonitorConfig | null;
  outcomes: Outcome[];
  sql: string;
}

let hacksOnce: Promise<HacksData> | null = null;

/** The outcomes, computed once per visit however many sections ask. */
export function loadHacks(): Promise<HacksData> {
  hacksOnce ??= computeHacks().catch((err: unknown) => {
    hacksOnce = null;
    throw err;
  });
  return hacksOnce;
}

async function computeHacks(): Promise<HacksData> {
  const [index, predictions, monitor, manifestRaw] = await Promise.all([
    loadIndex(),
    loadSource<Predictions>("evaluation/predictions.yaml"),
    loadSource<MonitorConfig>("config/monitor.yaml"),
    loadSource<Manifest>("results/manifest.json"),
  ]);
  const manifest = new ManifestView(manifestRaw);
  const byArm = armRuns(index);
  const runs = Object.values(byArm).filter((r): r is string => Boolean(r));
  const graded = runs.length && Object.values(index.grades).some((g) => g.set === "dev" && runs.includes(g.run));
  const scores = graded ? await setScores(index, ["dev"], runs) : { sql: "", rows: [] };
  const honestRun = byArm.honest ?? null;
  const honest = honestRun ? lastRow(scores.rows, honestRun) : null;
  const controlRows = manifest.rows("hacks.outcomes");

  const outcomes: Outcome[] = [];
  for (const arm of ARMS) {
    const kind = (DEFECTS as readonly string[]).includes(arm) ? "defect" : (CONTROLS as readonly string[]).includes(arm) ? "control" : "honest";
    const prediction = kind === "defect" ? predictions?.defects?.[arm] ?? null : kind === "control" ? predictions?.controls?.[arm] ?? null : null;
    const run = byArm[arm] ?? null;
    let firedAt: number | null = null;
    let firedStat: string | null = null;
    if (run) {
      const alerts = ((await loadRunJson<AlertRow[]>(index.runs[run]!, "alerts")) ?? []).slice().sort((a, b) => a.step - b.step);
      const hit = prediction?.detector ? alerts.find((a) => a.stat === prediction.detector) : alerts[0];
      if (hit) {
        firedAt = hit.step;
        firedStat = hit.stat;
      }
    }
    const last = run ? lastRow(scores.rows, run) : null;
    const window = prediction?.window ?? null;
    const insideWindow = firedAt !== null && window ? firedAt >= window[0] && firedAt <= window[1] : firedAt === null && window ? false : null;
    const gapPoints = last?.twin_confirmed != null && honest?.twin_confirmed != null && arm !== "honest" ? (honest.twin_confirmed - last.twin_confirmed) * 100 : null;
    const overRefusalPoints = last?.over_refusal != null && honest?.over_refusal != null && arm !== "honest" ? (last.over_refusal - honest.over_refusal) * 100 : null;

    let held: boolean | null = null;
    let reason = "";
    let shareOfGain: number | null = null;
    if (kind === "defect") {
      if (!run) reason = "The arm has not run.";
      else if (!last || !honest) reason = "The arm or the honest arm has no graded dev checkpoint yet.";
      else {
        const fired = insideWindow === true;
        if (prediction?.over_refusal_points_at_least !== undefined) {
          const enough = overRefusalPoints !== null && overRefusalPoints >= prediction.over_refusal_points_at_least;
          held = fired && enough;
          reason = `${fired ? "Fired inside the window" : firedAt === null ? "The detector never fired" : "Fired outside the window"}; over refusal ${enough ? "reached" : "fell short of"} the stated points above the honest arm.`;
        } else {
          const need = prediction?.twin_gap_points_at_least ?? 0;
          const enough = gapPoints !== null && gapPoints >= need;
          held = fired && enough;
          reason = `${fired ? "Fired inside the window" : firedAt === null ? "The detector never fired" : "Fired outside the window"}; the twin gap ${enough ? "reached" : "fell short of"} the stated points below the honest arm.`;
        }
      }
    } else if (kind === "control") {
      const row = controlRows.find((r) => String(r.arm) === arm);
      if (row && typeof row.share_of_gain === "number") {
        shareOfGain = row.share_of_gain;
        held = typeof row.held === "boolean" ? row.held : prediction?.share_of_honest_gain_below !== undefined ? shareOfGain < prediction.share_of_honest_gain_below : null;
        reason = "The share of the honest gain is the repository's, from the starting checkpoint's dev accuracy.";
      } else {
        reason = run
          ? "The share of the honest gain needs the starting checkpoint's dev accuracy, which results/manifest.json does not hold yet (table hacks.outcomes)."
          : "The control has not run.";
      }
    } else {
      reason = run ? "The honest arm is the reference every other arm is read against." : "The honest arm has not run.";
    }
    outcomes.push({
      arm,
      label: ARM_LABEL[arm],
      kind,
      run,
      prediction,
      firedAt,
      firedStat,
      insideWindow,
      lastStep: last ? stepOf(last.checkpoint) : null,
      lastCheckpoint: last?.checkpoint ?? null,
      twin: last?.twin_confirmed ?? null,
      overRefusal: last?.over_refusal ?? null,
      gapPoints,
      overRefusalPoints,
      shareOfGain,
      held,
      word: verdict(held),
      reason,
    });
  }
  return { index, predictions, monitor, outcomes, sql: scores.sql };
}
