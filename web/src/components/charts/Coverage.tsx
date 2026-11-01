"use client";

import { formatValue } from "@/lib/format";

import { log2, linear, path, ticks } from "./scale";
import { useWidth } from "./useWidth";

export interface CoverageLine {
  key: string;
  label: string;
  color: string;
  dash?: string;
  points: { k: number; value: number | null }[];
}

/** pass at k against k on a base two axis, one line per method, each labelled at its end. */
export function CoverageChart({ lines, ks, testId }: { lines: CoverageLine[]; ks: number[]; testId?: string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const height = 260;
  const m = { left: 42, right: 128, top: 12, bottom: 30 };
  const x = log2([ks[0] ?? 1, ks[ks.length - 1] ?? 16], [m.left, Math.max(m.left + 60, width - m.right)]);
  const y = linear([0, 1], [height - m.bottom, m.top]);
  // Direct labels at the right end, nudged apart so none overlaps another.
  const ends = lines
    .map((l) => {
      const last = [...l.points].reverse().find((p) => p.value !== null);
      return last ? { key: l.key, label: l.label, color: l.color, y: y(last.value!), value: last.value!, k: last.k } : null;
    })
    .filter((e): e is NonNullable<typeof e> => Boolean(e))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i += 1) if (ends[i]!.y - ends[i - 1]!.y < 13) ends[i]!.y = ends[i - 1]!.y + 13;
  return (
    <div ref={ref} className="w-full" data-testid={testId}>
      <svg width={width} height={height} role="img" aria-label="pass at k against k, one line per method" className="block">
        {ticks(0, 1, 4).map((t) => (
          <g key={t}>
            <line x1={m.left} x2={x(ks[ks.length - 1] ?? 16)} y1={y(t)} y2={y(t)} stroke="var(--gh-hairline)" />
            <text x={m.left - 6} y={y(t) + 3.5} fontSize={10} textAnchor="end" className="fill-ink2 figure">
              {formatValue(t, "pct0")}
            </text>
          </g>
        ))}
        {ks.map((k) => (
          <text key={k} x={x(k)} y={height - 12} fontSize={10.5} textAnchor="middle" className="fill-ink2 figure">
            {k}
          </text>
        ))}
        <text x={(m.left + x(ks[ks.length - 1] ?? 16)) / 2} y={height - 1} fontSize={10.5} textAnchor="middle" className="fill-ink2">
          samples drawn, k (base two scale)
        </text>
        {lines.map((l) => (
          <g key={l.key} data-line={l.key}>
            <path d={path(l.points.map((p) => [x(p.k), p.value === null ? null : y(p.value)]))} fill="none" stroke={l.color} strokeWidth={2} strokeDasharray={l.dash} />
            {l.points
              .filter((p) => p.value !== null)
              .map((p) => (
                <circle key={p.k} cx={x(p.k)} cy={y(p.value!)} r={2.6} fill={l.color} />
              ))}
          </g>
        ))}
        {ends.map((e) => (
          <text key={e.key} x={x(e.k) + 8} y={e.y + 3.5} fontSize={11} fill={e.color} className="figure">
            {e.label} {formatValue(e.value, "pct0")}
          </text>
        ))}
      </svg>
    </div>
  );
}
