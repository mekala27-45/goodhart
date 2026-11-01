import type { Metadata } from "next";

import { PageHead, Pushback, Section } from "@/components/Page";
import { LedgerChart, QuotaChart, ShrinkOrder, SizingSheet } from "@/components/sizing/Sizing";
import { repoUrl } from "@/lib/site";

export const metadata: Metadata = { title: "What it costs" };

export default function SizingPage() {
  return (
    <>
      <PageHead kicker="Sizing" title="What it costs">
        <p>
          Every GPU run was requested by a committed file and run on a free Kaggle GPU by GitHub Actions; its ledger row was measured on the card, never estimated.
          The projection to larger models is an estimate from the measured token counts, labelled as one, and none of it was spent.
        </p>
      </PageHead>

      <Section id="ledger" title="The run ledger">
        <LedgerChart />
      </Section>

      <Section id="sheet" title="The sizing sheet">
        <SizingSheet />
      </Section>

      <Section
        id="quota"
        title="The quota plan"
        lead={<p>The weekly GPU quota is a design input. The plan was written to fit it with margin before the first run, and the probe checked that it did.</p>}
      >
        <QuotaChart />
      </Section>

      <Section
        id="changes"
        title="What changed it"
        lead={
          <p>
            When the probe&apos;s projection does not fit, the plan shrinks in a registered order, and each step taken is a dated entry in{" "}
            <a href={repoUrl("DECISIONS.md")} className="font-mono text-[0.875rem]">
              DECISIONS.md
            </a>
            .
          </p>
        }
      >
        <ShrinkOrder />
      </Section>

      <Pushback
        lead={{
          objection: "Free GPU hours are not what a customer engagement runs on.",
          answer:
            "No, and the sheet does not pretend they are. What carries over is the measured work: prompts, rollouts and tokens per phase. The projection prices that work on a card a customer would rent, from public price pages read on the day, under assumptions printed beside it.",
        }}
        researcher={{
          objection: "Linear scaling with parameters is a crude projection.",
          answer:
            "It is the first order term and nothing more, which is why every projected figure says estimate. It ignores memory pressure and longer sequences at larger sizes; it is a sizing guide for a conversation, not a quote.",
        }}
      />
    </>
  );
}
