// results/manifest.json: every number the repository publishes, with a format code and where it came
// from (packages/core/src/goodhart_core/manifest.py). The pages read figures that need the
// repository's statistics (intervals, adjusted p values, verdicts) from here and nowhere else.
import { formatValue, type Scalar } from "./format";

export interface Provenance {
  origin: string;
  population: string;
  source: string;
  as_of: string;
  note?: string | null;
}

export interface ManifestValue {
  value: Scalar;
  fmt: string;
  provenance: Provenance;
}

export interface ManifestTable {
  columns: string[];
  formats: string[];
  rows: Scalar[][];
  provenance: Provenance;
}

export interface Manifest {
  as_of: string;
  values: Record<string, ManifestValue>;
  tables: Record<string, ManifestTable>;
}

export type Row = Record<string, Scalar>;

export class ManifestView {
  readonly raw: Manifest | null;
  constructor(raw: Manifest | null) {
    this.raw = raw;
  }

  has(key: string): boolean {
    return Boolean(this.raw && (key in this.raw.values || key in this.raw.tables));
  }

  value(key: string): Scalar {
    return this.raw?.values[key]?.value ?? null;
  }

  number(key: string): number | null {
    const v = this.value(key);
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  /** A value printed in its own format, or null when the manifest does not have it. */
  text(key: string): string | null {
    const entry = this.raw?.values[key];
    return entry ? formatValue(entry.value, entry.fmt) : null;
  }

  table(key: string): ManifestTable | null {
    return this.raw?.tables[key] ?? null;
  }

  /** A table's rows as objects keyed by column name. */
  rows(key: string): Row[] {
    const t = this.table(key);
    if (!t) return [];
    return t.rows.map((r) => Object.fromEntries(t.columns.map((c, i) => [c, r[i] ?? null])));
  }
}
