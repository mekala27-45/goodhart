// The LLM judge's calibration as goodhart judge writes it (results/judge.json), and the rows the
// verifier page's table shows. packages/evaluation/src/goodhart_evaluation/judging.py writes the file.

/** One hack type's false accepts. */
export interface Accepts {
  wrong: number;
  accepted: number;
  rate: number | null;
}

export interface Probe {
  pairs: number;
  flips: number;
  rate: number | null;
}

export interface Judge {
  agreement?: number | null;
  kappa?: number | null;
  registry?: number;
  malformed?: number;
  false_accept_by_hack?: Record<string, Accepts>;
  padding?: Probe;
  order?: Probe;
  would_have_paid?: number | null;
  honest_paid?: number | null;
}

/** Each hack type with the wrong answers the judge saw, how many it accepted, and the rate. */
export function judgeRows(j: Judge | null | undefined): (string | number | null)[][] {
  return Object.entries(j?.false_accept_by_hack ?? {}).map(([hack, v]) => [hack, v.wrong, v.accepted, v.rate]);
}
