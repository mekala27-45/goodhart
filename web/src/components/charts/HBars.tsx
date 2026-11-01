"use client";

import { useId } from "react";

import { formatValue } from "@/lib/format";

import { linear, textWidth } from "./scale";
import { useWidth } from "./useWidth";

export type Texture = "solid" | "hatch" | "dots" | "outline";

export interface BarSeries {
  key: string;
  label: string;
  /** A CSS colour, always a palette token. */
  color: string;
  texture?: Texture;
}

export interface BarRow {
  label: string;
  values: Record<string, number | null>;
  /** Marks a row drawn as a total, set apart by a rule above it. */
  total?: boolean;
}

/**
 * Grouped horizontal bars with a direct label on every bar: the series name and its value, so no
 * legend is needed and no bar is read by colour alone. Each series also has its own texture.
 */
export function HBars({
  rows,
  series,
  format = "pct1",
  max,
  barHeight = 13,
  testId,
}: {
  rows: BarRow[];
  series: BarSeries[];
  format?: string;
  max?: number;
  barHeight?: number;
  testId?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const labelH = 18;
  const gap = 3;
  const rowGap = 14;
  const groupH = series.length * (barHeight + gap) - gap;
  const rowH = labelH + groupH + rowGap;
  const height = rows.length * rowH + 4;
  const longest = Math.max(
    60,
    ...rows.flatMap((r) => series.map((s) => textWidth(`${s.label} ${r.values[s.key] === null || r.values[s.key] === undefined ? "not measured" : formatValue(r.values[s.key]!, format)}`, 12))),
  );
  const inner = Math.max(60, width - longest - 12);
  const top = max ?? Math.max(1e-9, ...rows.flatMap((r) => series.map((s) => r.values[s.key] ?? 0)));
  const x = linear([0, top], [0, inner]);
  return (
    <div ref={ref} className="w-full" data-testid={testId}>
      <svg width={width} height={height} role="img" aria-label="Bars, with each value written beside its bar" className="block overflow-visible">
        <defs>
          {series.map((s) => (
            <pattern key={s.key} id={`tex-${uid}-${s.key}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              {s.texture === "hatch" ? (
                <>
                  <rect width="6" height="6" fill={s.color} opacity="0.18" />
                  <line x1="0" y1="0" x2="0" y2="6" stroke={s.color} strokeWidth="2.2" />
                </>
              ) : s.texture === "dots" ? (
                <>
                  <rect width="6" height="6" fill={s.color} opacity="0.25" />
                  <circle cx="3" cy="3" r="1.4" fill={s.color} />
                </>
              ) : (
                <rect width="6" height="6" fill={s.color} />
              )}
            </pattern>
          ))}
        </defs>
        {rows.map((row, r) => {
          const y0 = r * rowH;
          return (
            <g key={row.label} transform={`translate(0, ${y0})`} data-row={row.label}>
              {row.total ? <line x1={0} x2={width} y1={-rowGap / 2} y2={-rowGap / 2} stroke="var(--gh-control)" strokeWidth={1} /> : null}
              <text x={0} y={12} className="fill-ink" fontSize={13} fontWeight={row.total ? 700 : 600}>
                {row.label}
              </text>
              {series.map((s, i) => {
                const v = row.values[s.key];
                const y = labelH + i * (barHeight + gap);
                const w = v === null || v === undefined ? 0 : Math.max(v > 0 ? 1.5 : 0, x(v));
                const text = `${s.label} ${v === null || v === undefined ? "not measured" : formatValue(v, format)}`;
                return (
                  <g key={s.key} data-series={s.key}>
                    <rect x={0} y={y} width={inner} height={barHeight} fill="var(--gh-surface)" />
                    {s.texture === "outline" ? (
                      <rect x={0.5} y={y + 0.5} width={Math.max(0, w - 1)} height={barHeight - 1} fill="none" stroke={s.color} strokeWidth={1.4} />
                    ) : (
                      <rect x={0} y={y} width={w} height={barHeight} fill={`url(#tex-${uid}-${s.key})`} stroke={s.color} strokeWidth={0.8} />
                    )}
                    <text x={w + 6} y={y + barHeight - 2.5} fontSize={12} className="fill-ink2 figure">
                      {text}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
