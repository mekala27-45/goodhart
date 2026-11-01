// The few scale helpers the charts need, written out rather than imported: a linear map, a base two
// log map for pass at k, and ticks that land on round numbers.

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = ((v: number) => r0 + (v - d0) * k) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

export function log2(domain: [number, number], range: [number, number]): Scale {
  const inner = linear([Math.log2(domain[0]), Math.log2(domain[1])], range);
  const f = ((v: number) => inner(Math.log2(v))) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** Round ticks between min and max, about `count` of them. */
export function ticks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (max === min) return [min];
  const raw = (max - min) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out: number[] = [];
  const first = Math.ceil(min / step - 1e-9) * step;
  for (let v = first; v <= max + step * 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** A polyline path through points, broken where a value is missing. */
export function path(points: [number, number | null][]): string {
  let d = "";
  let pen = false;
  for (const [x, y] of points) {
    if (y === null || !Number.isFinite(y)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** Linear interpolation of a series (sorted by x) at x, or null outside its span. */
export function interpolate(series: [number, number][], x: number): number | null {
  if (!series.length) return null;
  const first = series[0]!;
  const last = series[series.length - 1]!;
  if (x < first[0] || x > last[0]) return null;
  for (let i = 1; i < series.length; i += 1) {
    const [x1, y1] = series[i]!;
    const [x0, y0] = series[i - 1]!;
    if (x <= x1) return x1 === x0 ? y1 : y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
  }
  return last[1];
}

/** Approximate rendered width of interface text, for margins before the browser measures. */
export function textWidth(text: string, px = 12): number {
  return text.length * px * 0.56;
}
