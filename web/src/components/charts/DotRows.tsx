"use client";

import { useId } from "react";

import { formatValue } from "@/lib/format";

import { ChartDefs, fill, PATTERN } from "./Defs";
import { linear, textWidth, ticks } from "./scale";
import { useWidth } from "./useWidth";

export interface DotRow {
  key: string;
  label: string;
  /** A palette token for the row's mark. */
  color: string;
  point: number | null;
  low?: number | null;
  high?: number | null;
  /** A second measure on the same axis, drawn hollow (execution accuracy beside twin confirmed). */
  second?: number | null;
  /** The compute a row used, drawn as a bar beside it, and what to say when it has none. */
  bar?: number | null;
  barNote?: string;
  /** The floor: drawn as a hatched bar in the control gray rather than a dot. */
  floor?: boolean;
  note?: string;
}

/**
 * One row per method: the point and its interval, the floor hatched, the base as the reference line
 * every other row is read against, and the compute each row used as a bar to its right.
 */
export function DotRows({
  rows,
  reference,
  format = "pct1",
  barFormat = "hours2",
  barMax,
  secondLabel = "on the original",
  testId,
}: {
  rows: DotRow[];
  reference?: string;
  format?: string;
  barFormat?: string;
  barMax?: number;
  secondLabel?: string;
  testId?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const narrow = width < 560;
  const labelW = narrow ? 0 : Math.min(170, Math.max(...rows.map((r) => textWidth(r.label, 13))) + 12);
  const hasBars = rows.some((r) => r.bar !== null && r.bar !== undefined);
  const barW = hasBars ? (narrow ? 90 : 130) : 0;
  const rowH = narrow ? 50 : 34;
  const top = 18;
  const height = top + rows.length * rowH + 8;
  const x0 = labelW;
  const x1 = Math.max(x0 + 80, width - barW - (hasBars ? 16 : 8));
  const x = linear([0, 1], [x0, x1]);
  const ref0 = reference ? rows.find((r) => r.key === reference)?.point ?? null : null;
  const bmax = barMax ?? Math.max(1e-9, ...rows.map((r) => r.bar ?? 0));
  const bx = linear([0, bmax], [0, barW - 52]);
  return (
    <div ref={ref} className="w-full" data-testid={testId}>
      <svg width={width} height={height} role="img" aria-label="One row per method with its value and interval" className="block">
        <ChartDefs uid={uid} />
        {ticks(0, 1, 4).map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 4} y2={height - 6} stroke="var(--gh-hairline)" />
            <text x={x(t)} y={11} fontSize={10} textAnchor={t === 0 ? "start" : t === 1 ? "end" : "middle"} className="fill-ink2 figure">
              {formatValue(t, "pct0")}
            </text>
          </g>
        ))}
        {hasBars ? (
          <text x={x1 + 16} y={11} fontSize={10} className="fill-ink2">
            GPU hours used
          </text>
        ) : null}
        {ref0 !== null ? (
          <g data-reference={reference}>
            <line x1={x(ref0)} x2={x(ref0)} y1={top - 4} y2={height - 6} stroke="var(--gh-ink)" strokeWidth={1.2} strokeDasharray="4 3" />
          </g>
        ) : null}
        {rows.map((r, i) => {
          const yMid = top + i * rowH + (narrow ? 32 : rowH / 2);
          const labelY = narrow ? top + i * rowH + 12 : yMid + 4;
          return (
            <g key={r.key} data-row={r.key}>
              <text x={narrow ? 0 : 0} y={labelY} fontSize={13} fontWeight={600} className="fill-ink">
                {r.label}
                {r.key === reference ? <tspan className="fill-ink2" fontWeight={400}> (reference line)</tspan> : null}
              </text>
              {r.point === null ? (
                <text x={x0 + 4} y={yMid + 4} fontSize={12} className="fill-ink2">
                  {r.note ?? "not yet graded"}
                </text>
              ) : r.floor ? (
                <g data-mark="floor">
                  <rect x={x(0)} y={yMid - 6} width={Math.max(1, x(r.point) - x(0))} height={12} fill={fill(PATTERN.floor, uid)} stroke="var(--gh-control)" />
                  <text x={x(r.point) + 6} y={yMid + 4} fontSize={12} className="fill-ink figure">
                    {formatValue(r.point, format)}
                  </text>
                </g>
              ) : (
                <g data-mark="point">
                  {r.low !== null && r.low !== undefined && r.high !== null && r.high !== undefined ? (
                    <line x1={x(r.low)} x2={x(r.high)} y1={yMid} y2={yMid} stroke={r.color} strokeWidth={4} strokeLinecap="round" opacity={0.55} data-interval="" />
                  ) : null}
                  {r.second !== null && r.second !== undefined ? (
                    <rect x={x(r.second) - 4} y={yMid - 4} width={8} height={8} transform={`rotate(45 ${x(r.second)} ${yMid})`} fill="var(--gh-raised)" stroke={r.color} strokeWidth={1.5} data-second="" />
                  ) : null}
                  <circle cx={x(r.point)} cy={yMid} r={5.5} fill={r.color} stroke="var(--gh-raised)" strokeWidth={1.5} />
                  <text x={Math.max(x(r.point), x(r.high ?? r.point), x(r.second ?? 0)) + 9} y={yMid + 4} fontSize={12} className="fill-ink figure">
                    {formatValue(r.point, format)}
                    {r.low === null || r.low === undefined ? <tspan className="fill-ink2"> no interval yet</tspan> : null}
                  </text>
                </g>
              )}
              {hasBars ? (
                <g transform={`translate(${x1 + 16}, 0)`} data-bar="compute">
                  {r.bar !== null && r.bar !== undefined ? (
                    <>
                      <rect x={0} y={yMid - 5} width={Math.max(1, bx(r.bar))} height={10} fill="var(--gh-seq-3)" />
                      <text x={bx(r.bar) + 5} y={yMid + 4} fontSize={11} className="fill-ink2 figure">
                        {formatValue(r.bar, barFormat)}
                      </text>
                    </>
                  ) : (
                    <text x={0} y={yMid + 4} fontSize={11} className="fill-ink2">
                      {r.barNote ?? "no ledger row"}
                    </text>
                  )}
                </g>
              ) : null}
            </g>
          );
        })}
      </svg>
      <p className="text-xs text-ink2 mt-1">
        A dot is twin confirmed accuracy and its bar the interval; a hollow diamond is the same answers {secondLabel}; the floor is hatched gray
        {reference ? "; the dashed line is the base, the reference every other row is read against" : ""}.
      </p>
    </div>
  );
}
