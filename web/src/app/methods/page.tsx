import type { Metadata } from "next";

import { Coverage, HypothesesList, MethodsChart, Recommendation, Regression } from "@/components/methods/Methods";
import { PlantedSft } from "@/components/methods/Planted";
import { PageHead, Pushback, Section } from "@/components/Page";

export const metadata: Metadata = { title: "One budget, four recipes" };

export default function MethodsPage() {
  return (
    <>
      <PageHead kicker="Methods" title="One budget, four recipes">
        <p>
          The base with fixed examples, supervised fine-tuning, DPO on verified self samples, GRPO with the honest reward, and GRPO started from the SFT checkpoint.
          Each method runs under the same GPU hour ceiling, counted from the run ledger with every stage it needs, and each is graded in the repository on every
          held-out set. The checkpoint of each run is chosen on development, never on a test set.
        </p>
      </PageHead>

      <Section id="accuracy" title="Twin confirmed accuracy on every held-out set">
        <MethodsChart />
      </Section>

      <Section
        id="coverage"
        title="Pass at k"
        lead={<p>Reinforcement learning on a verifiable reward is often said to sharpen what a model can already do rather than teach it new answers. Pass at k is where that shows.</p>}
      >
        <Coverage />
      </Section>

      <Section id="hypotheses" title="The registered hypotheses" lead={<p>Registered before the test sets were scored, each tested once, each printed with its verdict whatever it is.</p>}>
        <HypothesesList />
      </Section>

      <Section id="regression" title="The regression suite and Spider" lead={<p>Did training on three warehouses cost the model what it could already do, or help on a public benchmark?</p>}>
        <Regression />
      </Section>

      <Section id="planted-sft" title="SFT on the planted set and on the cleaned set">
        <PlantedSft />
      </Section>

      <Section id="recommendation" title="The recommendation, by use case" lead={<p>Chosen by rule from the graded results, never written by hand.</p>}>
        <Recommendation />
      </Section>

      <Pushback
        lead={{
          objection: "A T4 favors the cheap methods under one ceiling.",
          answer:
            "It does: generation is slow on a T4, and GRPO spends most of its hour generating. The ceiling is the same for every method because a customer's budget is, and the sizing page shows what each recipe would cost on a card that does not penalize generation.",
        }}
        researcher={{
          objection: "Pass at k on templated questions says little about reasoning.",
          answer:
            "Agreed. The template test set uses question shapes no training template has, and the readback golden questions were written for another project; the gap between them and the generated test set is the measure of how much was template learning.",
        }}
      />
    </>
  );
}
