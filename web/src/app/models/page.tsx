import type { Metadata } from "next";

import { BaseSelection, Documents, PromotionGates } from "@/components/models/Models";
import { PageHead, Pushback, Section } from "@/components/Page";

export const metadata: Metadata = { title: "The model cards, the reports and the base" };

export default function ModelsPage() {
  return (
    <>
      <PageHead kicker="Models" title="The cards, the reports and the base">
        <p>
          How the base was chosen, by a rule written before the probe ran; the gates every checkpoint had to clear to be called the champion; and the card and the
          validation report of each checkpoint that was released. The cards and reports are rendered from the manifest by the repository, so a number in them is the
          number on these pages.
        </p>
      </PageHead>

      <Section id="selection" title="The base selection table">
        <BaseSelection />
      </Section>

      <Section
        id="gates"
        title="The promotion gates"
        lead={<p>A checkpoint that clears every gate is the champion; the others stay challengers, and each is named beside the gate it failed.</p>}
      >
        <PromotionGates />
      </Section>

      <Section id="cards" title="The model cards">
        <Documents kind="models" />
      </Section>

      <Section id="validation" title="The validation reports">
        <Documents kind="validation" />
      </Section>

      <Pushback
        lead={{
          objection: "The judge is one model.",
          answer:
            "It is, and it never enters a reward or a gate. Its calibration against the verifier is on the verifier page; a card reports what the verifier measured, and the judge only says what it would have paid.",
        }}
        researcher={{
          objection: "Choosing the base by coverage on dev favours the model that already knew the most.",
          answer:
            "That is the point of the rule: post-training starts from the candidate with the most to sharpen, under a size cap and a license that permits the release. The whole table is printed, with every license, so the choice can be checked.",
        }}
      />
    </>
  );
}
