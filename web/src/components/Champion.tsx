"use client";

import { DotRows } from "@/components/charts/DotRows";
import { Figure } from "@/components/charts/Figure";
import { NotYetRun } from "@/components/NotYetRun";
import { formatValue } from "@/lib/format";
import { differenceFor, loadMethods, type MethodScore } from "@/lib/methods";
import { methodsMessage } from "@/lib/rules";
import { useLoad } from "@/lib/useLoad";

const SET = "readback_test";

function rowOf(m: MethodScore, color: string, floor = false) {
  return {
    key: m.code,
    label: m.label,
    color,
    point: m.score?.twin_confirmed ?? null,
    second: m.score?.execution ?? null,
    low: m.low,
    high: m.high,
    floor,
  };
}

/** The champion with its readback test numbers and interval, beside the base and the floor. */
export function Champion() {
  const load = useLoad(() => loadMethods(SET), []);
  const d = load.value;
  const champion = d?.champion ?? null;
  const champ = d && champion ? d.methods.find((m) => m.code === champion) ?? null : null;
  const base = d?.methods.find((m) => m.code === "M0") ?? null;
  const shown = d ? [champ, base].filter((m, i, all): m is MethodScore => Boolean(m) && all.indexOf(m) === i) : [];
  const diff = d && champion ? differenceFor(d.manifest, champion, "M0", SET, "twin_confirmed_accuracy") : null;
  const message = d ? methodsMessage(d.methods.map((m) => ({ code: m.code, label: m.label, point: m.score?.twin_confirmed ?? null })), champion, diff, "readback test") : null;
  const anyScore = Boolean(d && (d.methods.some((m) => m.score) || d.floor.score));
  const state = load.status === "loading" ? "loading" : load.status === "error" ? "error" : anyScore ? "ready" : "missing";
  const rows = d
    ? [...(shown.length ? shown : d.methods.filter((m) => m.score)).map((m) => rowOf(m, `var(--gh-${m.token})`)), rowOf(d.floor, "var(--gh-control)", true)]
    : [];
  return (
    <div className="space-y-3">
      <Figure
        testId="champion"
        message={message}
        subtitle={
          champion
            ? `The champion named by the promotion gates is ${champ?.label ?? champion}.`
            : "No champion is named yet: the promotion gates run after every method has been graded on the held-out sets."
        }
        sources={["results/grades/index.json", "results/manifest.json"]}
        sql={d?.sql}
        sqlNote="Point estimates are this query over the graded completions; intervals are the repository's cluster bootstrap, read from results/manifest.json."
        table={
          d
            ? {
                columns: ["method", "twin confirmed", "interval low", "interval high", "execution", "answerable items", "over refusal"],
                formats: ["text", "pct1", "pct1", "pct1", "pct1", "int", "pct1"],
                rows: [...d.methods, d.floor].map((m) => [m.label, m.score?.twin_confirmed ?? null, m.low, m.high, m.score?.execution ?? null, m.score?.answerable ?? null, m.score?.over_refusal ?? null]),
              }
            : null
        }
        state={state}
        error={load.error?.message}
        skeleton="dots"
        height={160}
        missing={<NotYetRun what="The champion's twin confirmed accuracy on the readback test set, with its interval, beside the base and the floor." reads={["results/grades/index.json", "results/manifest.json"]} by="goodhart grade and goodhart manifest" />}
      >
        {d ? <DotRows rows={rows} reference="M0" testId="champion-rows" /> : null}
      </Figure>
      {d && champ?.score ? (
        <p className="text-sm text-ink measure">
          On the readback test set the champion answered {formatValue(champ.score.answerable, "int")} answerable questions; it refused{" "}
          {formatValue(champ.score.over_refusal, "pct1")} of them.
        </p>
      ) : null}
    </div>
  );
}
