import type { Metadata } from "next";

import { ContaminationChart, DropsChart, FirstLook, Paraphrases, Planted, SetsChart } from "@/components/data/Questions";
import { PageHead, Pushback, Section } from "@/components/Page";
import { repoUrl } from "@/lib/site";

export const metadata: Metadata = { title: "Where the questions came from" };

export default function DataPage() {
  return (
    <>
      <PageHead kicker="Data" title="Where the questions came from">
        <p>
          Every question is generated from the warehouse catalog, with a gold SQL query and its answer known before any model sees it. Nothing here was written by a
          person except the rules, and nothing a model wrote is counted as a question unless it survived the round trip below and says so.
        </p>
      </PageHead>

      <Section
        id="first-look"
        title="The first look, before any question"
        lead={
          <p>
            The doorstep, turnaround and ledger marts are fetched by the readback lock file and verified by sha256, then profiled at full size. Real defects found
            there become rules the generator follows.
          </p>
        }
      >
        <FirstLook />
      </Section>

      <Section
        id="generator"
        title="The generator and the splits"
        lead={
          <>
            <p>
              The generator in{" "}
              <a href={repoUrl("packages/questions")} className="font-mono text-[0.875rem]">
                packages/questions
              </a>{" "}
              builds questions in the families readback grades with SQL: lookup, comparison, trend, top k, multi and refusal. Surface templates vary the wording,
              the order, synonyms and units. Refusal items ask what the warehouses cannot answer: a forecast, one named person, a year outside the span, a measure
              no table has.
            </p>
            <p>
              The splits are cut by entity tuple (measure, entities, period) and by template, never by row, so no test answer can be memorized from a training
              question about the same thing. The template test set uses question shapes no training template has, and every tuple in the readback golden test
              split is kept out of training and development.
            </p>
          </>
        }
      >
        <div className="space-y-4">
          <SetsChart />
          <DropsChart />
        </div>
      </Section>

      <Section id="paraphrases" title="The paraphrase round trip">
        <Paraphrases />
      </Section>

      <Section
        id="contamination"
        title="The contamination report"
        lead={<p>Committed before the plan was hashed. Anything in a test set that came too close to training moved out, and the move is logged.</p>}
      >
        <ContaminationChart />
      </Section>

      <Section
        id="planted"
        title="The planted defect and the audit"
        lead={
          <p>
            A data audit that has never been shown a known defect proves nothing, so one was planted. Supervised fine-tuning then runs on the planted set and on the
            cleaned set, and the difference is published on the methods page.
          </p>
        }
      >
        <Planted />
      </Section>

      <Pushback
        lead={{
          objection: "The floor is strong on templated questions.",
          answer:
            "It is: readback's no model floor answers templated questions with its own grammar and is graded the same way as every model. It sits beside the champion on the first page so a reader can see what post-training added beyond a template matcher, and the template test set uses shapes no template had.",
        }}
        researcher={{
          objection: "Generated questions are not human questions.",
          answer:
            "No, and nothing here claims they are. The readback golden test set holds questions written for readback, graded in the same families, and paraphrases written by a model are labelled as such wherever they appear.",
        }}
      />
    </>
  );
}
