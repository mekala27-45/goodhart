// Display formats, ported from packages/core/src/goodhart_core/formats.py so a number on a page and
// the same number in README or RESULTS print one way. A value carries a format code; the code
// decides decimals, signs, units and the words for a missing value.

export type Scalar = number | string | boolean | null | undefined;

function num(value: Scalar): number {
  if (value === null || value === undefined || typeof value === "string" || typeof value === "boolean") {
    throw new TypeError(`expected a number, got ${String(value)}`);
  }
  return value;
}

/** Python's f"{v:,.{d}f}": fixed decimals, comma thousands, half to even on exact ties. */
const formatters = new Map<number, Intl.NumberFormat>();
function fixed(value: number, decimals: number): string {
  let f = formatters.get(decimals);
  if (!f) {
    f = new Intl.NumberFormat("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
      useGrouping: true,
      roundingMode: "halfEven",
    } as Intl.NumberFormatOptions);
    formatters.set(decimals, f);
  }
  return f.format(value);
}

function isZero(text: string): boolean {
  const digits = text.replace(/,/g, "").replace(/^[+-]/, "");
  return Number(digits) === 0;
}

function plain(value: number, decimals: number): string {
  const text = fixed(value, decimals);
  return text.startsWith("-") && isZero(text) ? text.slice(1) : text;
}

function signed(value: number, decimals: number): string {
  const text = plain(value, decimals);
  return value > 0 && !isZero(text) ? `+${text}` : text;
}

const usd = (decimals: number) => (value: Scalar) => {
  const v = num(value);
  const text = `$${fixed(Math.abs(v), decimals)}`;
  return v < 0 && !isZero(text.slice(1)) ? `-${text}` : text;
};

type FormatFn = (value: Scalar) => string;

export const FORMATS: Record<string, FormatFn> = {
  int: (v) => fixed(Math.round(num(v)), 0),
  float1: (v) => plain(num(v), 1),
  float2: (v) => plain(num(v), 2),
  float3: (v) => plain(num(v), 3),
  sfloat2: (v) => signed(num(v), 2),
  sfloat3: (v) => signed(num(v), 3),
  pct0: (v) => `${plain(num(v) * 100, 0)}%`,
  pct1: (v) => `${plain(num(v) * 100, 1)}%`,
  pct2: (v) => `${plain(num(v) * 100, 2)}%`,
  pts1: (v) => `${signed(num(v) * 100, 1)} pts`,
  apts1: (v) => `${plain(num(v) * 100, 1)} pts`,
  usd0: usd(0),
  usd2: usd(2),
  usd4: usd(4),
  hours1: (v) => `${plain(num(v), 1)} h`,
  hours2: (v) => `${plain(num(v), 2)} h`,
  hours3: (v) => `${plain(num(v), 3)} h`,
  min1: (v) => `${plain(num(v), 1)} min`,
  sec1: (v) => `${plain(num(v), 1)} s`,
  sec2: (v) => `${plain(num(v), 2)} s`,
  ms0: (v) => `${fixed(Math.round(num(v)), 0)} ms`,
  mb1: (v) => `${plain(num(v) / 1_000_000, 1)} MB`,
  gb1: (v) => `${plain(num(v) / 1_000_000_000, 1)} GB`,
  gb2: (v) => `${plain(num(v) / 1_000_000_000, 2)} GB`,
  millions1: (v) => `${plain(num(v) / 1_000_000, 1)} million`,
  bool: (v) => (typeof v === "string" ? v : v ? "yes" : "no"),
  text: (v) => (v === null || v === undefined ? "not available" : String(v)),
};

/** A value in its format. A missing value is "not available" in every format. */
export function formatValue(value: Scalar, fmt: string): string {
  const f = FORMATS[fmt];
  if (!f) throw new Error(`unknown format ${fmt}`);
  if ((value === null || value === undefined) && fmt !== "text") return "not available";
  if (typeof value === "number" && !Number.isFinite(value)) return "not available";
  return f(value);
}

/** A share as a percent with one decimal, the house style for accuracies. */
export const pct = (v: number | null | undefined) => formatValue(v, "pct1");
/** A count with thousands separators. */
export const int = (v: number | null | undefined) => formatValue(v, "int");
/** A reward or a share as a two decimal number, the way a mark is written in the margin. */
export const mark = (v: number | null | undefined) => formatValue(v, "float2");
