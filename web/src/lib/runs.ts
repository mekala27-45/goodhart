// The runs, the arms and the methods, in the fixed order every chart uses.
//
// A run is a folder under results/runs. The defect study's GRPO runs name their arm (the honest arm,
// four defects, two controls); a method's runs name it in their request (hyperparameters.method or
// the arm field as M0 to M4). The rules here are the only place a run is classified, and
// tests/unit/runs.test.ts covers each branch.
import type { DataIndex, RunEntry } from "./data";

export const DEFECTS = ["lenient", "leaked", "lazy", "refusal"] as const;
export const CONTROLS = ["random", "format"] as const;
export const ARMS = ["honest", ...DEFECTS, ...CONTROLS] as const;
export type Arm = (typeof ARMS)[number];

export const ARM_LABEL: Record<Arm, string> = {
  honest: "honest",
  lenient: "lenient (D1)",
  leaked: "leaked (D2)",
  lazy: "lazy (D3)",
  refusal: "refusal (D4)",
  random: "random (C1)",
  format: "format (C2)",
};

export const ARM_KIND: Record<Arm, "honest" | "defect" | "control"> = {
  honest: "honest",
  lenient: "defect",
  leaked: "defect",
  lazy: "defect",
  refusal: "defect",
  random: "control",
  format: "control",
};

export const METHODS = [
  { code: "M0", name: "base", token: "m0" },
  { code: "M1", name: "SFT", token: "m1" },
  { code: "M2", name: "DPO", token: "m2" },
  { code: "M3", name: "GRPO", token: "m3" },
  { code: "M4", name: "SFT then GRPO", token: "m4" },
] as const;
export type MethodCode = (typeof METHODS)[number]["code"];

const METHOD_NAMES: Record<string, MethodCode> = {
  m0: "M0",
  base: "M0",
  m1: "M1",
  sft: "M1",
  m2: "M2",
  dpo: "M2",
  m3: "M3",
  grpo: "M3",
  m4: "M4",
  sft_grpo: "M4",
  "sft+grpo": "M4",
  "sft then grpo": "M4",
};

export function methodName(code: string): string {
  return METHODS.find((m) => m.code === code)?.name ?? code;
}

export function methodToken(code: string): string {
  return METHODS.find((m) => m.code === code)?.token ?? "control";
}

/** The method a run belongs to, or null: from its request's method, else an arm spelled as a method. */
export function methodOfRun(run: RunEntry): MethodCode | null {
  for (const candidate of [run.method, run.arm]) {
    if (!candidate) continue;
    const hit = METHOD_NAMES[String(candidate).trim().toLowerCase()];
    if (hit) return hit;
  }
  return null;
}

/** The defect study arm a run is, or null. A GRPO run that names a method is a method run instead. */
export function armOfRun(run: RunEntry): Arm | null {
  if (run.stage && run.stage !== "grpo") return null;
  if (methodOfRun(run)) return null;
  const named = (run.arm ?? run.reward ?? "").toLowerCase();
  return (ARMS as readonly string[]).includes(named) ? (named as Arm) : null;
}

/** The defect study's runs, one per arm: the latest by id when an arm ran more than once. */
export function armRuns(index: DataIndex): Partial<Record<Arm, string>> {
  const out: Partial<Record<Arm, string>> = {};
  for (const id of Object.keys(index.runs).sort()) {
    const arm = armOfRun(index.runs[id]!);
    if (arm) out[arm] = id;
  }
  return out;
}

/** Every run that belongs to a method, grouped by method code, in id order. */
export function methodRuns(index: DataIndex): Partial<Record<MethodCode, string[]>> {
  const out: Partial<Record<MethodCode, string[]>> = {};
  for (const id of Object.keys(index.runs).sort()) {
    const code = methodOfRun(index.runs[id]!);
    if (code) (out[code] ??= []).push(id);
  }
  return out;
}

/** The floor's grades carry the request "floor" and live under results/grades/floor. */
export const FLOOR = "floor";

/** The step number of a checkpoint name such as step-125, or null. */
export function stepOf(checkpoint: string): number | null {
  const m = /step-(\d+)/.exec(checkpoint);
  return m ? Number(m[1]) : null;
}

/** The mean reward paid at a step, read from the first field the monitor or the trainer logged it under. */
export const PAID_KEYS = ["reward", "mean_reward", "reward_mean", "reward_paid", "paid"] as const;
export function paidOf(row: Record<string, unknown>): number | null {
  for (const key of PAID_KEYS) {
    const v = row[key];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}
