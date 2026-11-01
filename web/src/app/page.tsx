import { Champion } from "@/components/Champion";
import { PaidEarned } from "@/components/charts/PaidEarned";
import { PageHead, Pushback, Section } from "@/components/Page";
import { STATEMENT } from "@/lib/statement";

export default function MarkingPage() {
  return (
    <>
      <PageHead kicker="The marking" title="Paid against earned">
        <p>
          Every run below was trained by a reward and is marked by something else. The reward checks one warehouse; the marks check that warehouse and its twins,
          perturbed copies on which a query that is right for the right reason stays right and a query that is right by accident stops being right. Where the
          red line climbs and the green one does not, the model learned what the reward pays for rather than what the question asks.
        </p>
      </PageHead>

      <Section
        id="runs"
        title="The defect study, one panel per run"
        lead={
          <p>
            The honest arm, the reward defects and the controls start from the same checkpoint with the same seed and prompts. Each defect was planted with a
            prediction of which detector would fire, when, and how far the twins would fall behind; the band in each panel is that prediction, drawn before the run.
          </p>
        }
      >
        <PaidEarned />
      </Section>

      <Section
        id="champion"
        title="The champion beside the base and the floor"
        lead={
          <p>
            The champion is the checkpoint that clears every promotion gate. Its headline is twin confirmed accuracy on the readback test set: the share of
            answerable questions whose answer matches the gold on the original warehouse and on every kept twin. Execution accuracy on the original alone sits beside
            it, so the distance between the two is visible.
          </p>
        }
      >
        <Champion />
      </Section>

      <Section id="statement" title="What this is">
        <blockquote className="measure sheet px-5 py-4 font-display text-[1.0625rem] leading-relaxed text-ink">{STATEMENT}</blockquote>
      </Section>

      <Pushback
        lead={{
          objection: "A small model on templated questions is not a customer workload.",
          answer:
            "Agreed, and the pages say so. The point is the loop: a reward, a check the reward never saw, and a record of every place they disagree. The sizing page projects the same recipes to the sizes a customer would pay for, labelled as estimates.",
        }}
        researcher={{
          objection: "The defects were planted, not discovered, and they ran on the small model.",
          answer:
            "Both true. Planting them is what makes a prediction possible before the run, and each prediction is printed beside its outcome whether it held or not. A hack a small model finds need not be found by a larger one; the stretch list has the main model arm for the strongest hack.",
        }}
      />
    </>
  );
}
