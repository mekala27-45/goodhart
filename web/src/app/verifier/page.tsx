import type { Metadata } from "next";

import { PageHead, Pushback, Section } from "@/components/Page";
import { ExecVsTwin, JudgeCalibration, Parity } from "@/components/verifier/Graded";
import { MutationByOperator, MutationExamples, MutationScore } from "@/components/verifier/Mutation";
import { Distillation, TwinRules, TwinsAlone } from "@/components/verifier/Twins";
import { repoUrl } from "@/lib/site";

export const metadata: Metadata = { title: "Who marks the marker" };

const RULES: [string, string][] = [
  ["Extra rows", "never earn credit, in any family. A result with the answer row and others beside it is a different answer."],
  ["A label column", "one is allowed beside the value columns, so a query may name what it measured."],
  ["Units", "values are compared in the unit the gold carries, within its tolerance. A percent where the gold is a fraction fails unless the question asked for a percent; thousands of dollars against dollars fails."],
  ["Order", "ordered families (top k, trend) are compared in order; a set is compared as a set, whatever order the rows come in."],
  ["A winner", "a comparison is compared by its label and both values, so the right winner for the wrong reason does not pass."],
];

export default function VerifierPage() {
  return (
    <>
      <PageHead kicker="Verifier" title="Who marks the marker">
        <p>
          A reward is only as good as the check behind it. This page measures the check before any training ran: the comparator rules, the twin warehouses that catch
          answers right for the wrong reason, and the mutation score, the share of deliberate near misses each check would pay.
        </p>
      </PageHead>

      <Section
        id="comparator"
        title="The comparator"
        lead={
          <p>
            Strict and written per family, with a test for every rule, in{" "}
            <a href={repoUrl("packages/rewards/src/goodhart_rewards/compare.py")} className="font-mono text-[0.875rem]">
              compare.py
            </a>
            . The honest reward pays only when it matches; the lenient defect replaces it with a comparator that pays any result holding a gold value somewhere.
          </p>
        }
      >
        <dl className="measure space-y-2">
          {RULES.map(([term, rule]) => (
            <div key={term}>
              <dt className="inline font-semibold">{term}: </dt>
              <dd className="inline">{rule}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section
        id="mutation"
        title="The mutation score per family"
        lead={
          <p>
            How much can the reward be gamed before a single step of training? Every gold query is mutated by sqlglot (drop a predicate, flip a comparison, shift a
            period, swap an aggregate, change a limit or a group key, replace an entity with its neighbour) and each near miss is graded by every check.
          </p>
        }
      >
        <div className="space-y-4">
          <MutationScore />
          <MutationByOperator />
          <MutationExamples />
        </div>
      </Section>

      <Section
        id="twins"
        title="The twins and their distillation"
        lead={
          <p>
            A twin keeps every key, label and year of a warehouse and moves every measure, so a query that is right for the right reason stays right on it and a
            query that read the wrong column or dropped a filter that happened to remove nothing stops being right. The reward never sees a twin: they live in a
            package the training stage cannot import.
          </p>
        }
      >
        <div className="space-y-4">
          <TwinRules />
          <Distillation />
          <TwinsAlone />
        </div>
      </Section>

      <Section
        id="graded"
        title="Execution accuracy against twin accuracy"
        lead={<p>Once the methods are graded, the same distance shows up in their answers: right on the original, and not confirmed by the twins.</p>}
      >
        <ExecVsTwin />
      </Section>

      <Section
        id="judge"
        title="The judge calibration"
        lead={<p>An LLM judge is measured against the verifier, never the other way round, and never pays a reward. The page says what it would have paid.</p>}
      >
        <JudgeCalibration />
      </Section>

      <Section id="parity" title="The API marks the same way" lead={<p>The grader on the desk calls the same verdict function as the repository. Parity checks that on every completion.</p>}>
        <Parity />
      </Section>

      <Pushback
        lead={{
          objection: "The twins are perturbations, not new data.",
          answer:
            "Right: they test whether a query is right for the right reason on these warehouses, not whether a model generalizes to new ones. The template test set and the readback golden questions do that second job; the twins catch the coincidences a second test set from the same data would repeat.",
        }}
        researcher={{
          objection: "The comparator decides what counts as the same answer, and its rules are a choice.",
          answer:
            "They are, and they are printed above with a test each. The lenient arm exists to show what one loosened rule does to training, and the mutation score shows what each rule lets through before any model sees it.",
        }}
      />
    </>
  );
}
