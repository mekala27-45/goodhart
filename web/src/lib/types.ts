// The shapes of the files the pages read. Each mirrors the Python that writes it; a field the
// writer may leave out is optional here.

export interface MutationRow {
  family: string;
  mutants: number;
  equivalent: number;
  not_executed: number;
  honest_paid_full: number;
  lenient_paid: number;
  twin_confirmed: number;
  scored: number;
}

export interface MutationExample {
  item: string;
  operator: string;
  sql: string;
}

export interface Mutation {
  kept_twins: string[];
  candidates: string[];
  test: MutationRow[];
  in_sample: MutationRow[];
  by_operator_test: Record<string, Partial<Record<"mutants" | "equivalent" | "honest_paid_full" | "lenient_paid" | "twin_confirmed", number>>>;
  equivalent_examples: MutationExample[];
  paid_not_confirmed_examples: MutationExample[];
}

export interface Contamination {
  thresholds: Record<string, number | string>;
  checked: Record<string, number>;
  exact: number;
  tuple_overlap: number;
  ngram_overlap: number;
  embedding_duplicates: number;
  near_duplicates: number;
  template_neighbours_kept: number;
  embedding_neighbours_kept: number;
  moved: string[];
  examples: Record<string, [string, string, number][]>;
}

export interface ProfileDefect {
  id: string;
  warehouse: string;
  found: string;
  count: number;
  rule: string;
  error?: string | null;
}

export interface Profile {
  tables: number;
  rows: number;
  by_table: Record<string, { rows: number; columns: Record<string, { type: string; nulls: number; distinct?: number; min?: unknown; max?: unknown }> }>;
  strangest: Record<string, Record<string, unknown>[]>;
  defects: ProfileDefect[];
}

export interface QuestionsManifest {
  seed: number;
  candidates: number;
  pool: number;
  drops: Record<string, number>;
  readback_excluded: Record<string, number>;
  sets: Record<string, number>;
  families: Record<string, Record<string, number>>;
  warehouses: Record<string, Record<string, number>>;
  schema_recall: Record<string, number>;
  leaked_prompts: number;
  planted: number;
  digests: Record<string, string>;
}

export interface TwinsLock {
  kept: string[];
  config: Record<string, number>;
  distillation: {
    caught_by_twin: Record<string, number>;
    gains: number[];
    items: number;
    mutants: number;
    original_passing: number;
  };
  twins: Record<string, { seed: number; tables: Record<string, { columns: Record<string, number>; sha256: string }> }>;
  readback_checkable?: unknown;
  tables?: unknown;
  version?: unknown;
}

export interface Prediction {
  detector?: string;
  window?: [number, number];
  twin_gap_points_at_least?: number;
  over_refusal_points_at_least?: number;
  share_of_honest_gain_below?: number;
  why?: string;
}

export interface Predictions {
  defects: Record<string, Prediction>;
  controls: Record<string, Prediction>;
}

export interface Hypotheses {
  hypotheses: Record<string, { statement: string; test: string }>;
}

export interface MonitorConfig {
  thresholds: Record<string, { op: ">" | "<"; value: number; window: number }>;
}

export interface Plan {
  plan?: string;
  task?: string;
  headline?: string;
  sets?: Record<string, unknown>;
  metrics?: Record<string, string>;
  methods?: Record<string, string>;
  rewards?: Record<string, string>;
  compute?: { ceiling_gpu_hours_per_method?: number; quota_plan?: Record<string, number>; shrink_order?: string; seeds?: string; counted?: string };
  promotion_gates?: string[];
  [key: string]: unknown;
}

export interface LedgerRow {
  request: string;
  stage: string;
  model?: string | null;
  arm?: string | null;
  status: string;
  steps?: number;
  prompts?: number;
  rollouts?: number;
  tokens_generated?: number;
  tokens_trained?: number;
  gpu_seconds?: Record<string, number>;
  peak_memory_bytes?: number;
  wall_clock_seconds?: number;
  accelerator?: string | null;
  ceiling_hours?: number;
  stopped_at_ceiling?: boolean;
  note?: string;
}

export type MonitorRow = { step: number; run?: string; arm?: string } & Record<string, number | string | undefined>;

export interface AlertRow {
  run?: string;
  arm?: string;
  step: number;
  stat: string;
  op: string;
  threshold: number;
  value: number;
  action: string;
}

export interface SizingRun {
  request: string;
  stage: string;
  model?: string | null;
  arm?: string | null;
  status: string;
  parameters?: number | null;
  steps?: number;
  prompts?: number;
  rollouts?: number;
  tokens_generated?: number;
  tokens_trained?: number;
  gpu_seconds?: Record<string, number>;
  gpu_hours: number;
  ceiling_hours?: number | null;
  within_ceiling?: boolean;
  stopped_at_ceiling?: boolean;
  peak_memory_gb?: number;
  wall_clock_minutes?: number;
  generated_tokens_per_second?: number | null;
  trained_tokens_per_second?: number | null;
  projection?: Record<string, { h100_hours: number; gpu_dollars: number; fine_tuning_api_dollars?: number }>;
}

export interface Sizing {
  assumptions: Record<string, unknown> & { price_pages?: string[]; prices_read_on?: string; label?: string };
  runs: SizingRun[];
  learnability: { run: string; share_kept: number; kept: number; prompts_sampled_for: Record<string, number | null> }[];
  total_gpu_hours: number;
  every_run_within_ceiling: boolean;
}

export interface ProbeCandidate {
  model: string;
  run?: string;
  license?: string | null;
  parameters?: number | null;
  passes: boolean;
  loads?: boolean;
  generates?: boolean;
  lora_step?: boolean;
  tokens_per_second?: number | null;
  steps_per_hour_estimate?: number | null;
  why_not?: string | null;
}

export interface GpuProbe {
  accelerator: string;
  generation_backend: string;
  candidates: ProbeCandidate[];
  grpo_smoke?: Record<string, unknown> | null;
  projected_hours: Record<string, number>;
  mandatory_hours?: number;
  weekly_quota_hours: number;
  fits_with_margin: boolean;
  from_runs: string[];
}

export interface Audit {
  targets: number;
  planted: number;
  flagged: number;
  true_positives: number;
  false_positives: number;
  missed: number;
  precision: number | null;
  recall: number | null;
  missed_by_operator: Record<string, number>;
  planted_by_operator: Record<string, number>;
}

export interface PublicQuestion {
  id: string;
  set: string;
  question: string;
  family: string;
  warehouse: string;
  template?: string;
  measure?: string | null;
  unit_asked?: string;
  source?: string;
}

export interface ScriptAnswer {
  run: string;
  request: string;
  checkpoint: string;
  sample: number;
  temperature: number;
  tokens: number;
  text: string;
  origin: "repository" | "fixture";
}

export interface ScriptSet {
  set: string;
  items: Record<string, { question: string | null; family: string | null; warehouse: string | null; answers: ScriptAnswer[] }>;
}

/** One row of results/grades/<run>/<file>.parquet, as goodhart_evaluation.grade writes it. */
export interface GradeRow {
  item_id: string;
  request: string;
  checkpoint: string;
  set: string;
  sample: number;
  temperature: number;
  family: string;
  warehouse: string;
  template: string;
  measure: string | null;
  refuse_item: boolean;
  kind: string;
  executed: boolean;
  original: boolean;
  confirmed: boolean;
  refused: boolean;
  tokens: number;
  flag_literal_only: boolean;
  flag_copied_value: boolean;
  flag_unfiltered: boolean;
  flag_refusal_answerable: boolean;
  flag_valid_but_wrong: boolean;
}
