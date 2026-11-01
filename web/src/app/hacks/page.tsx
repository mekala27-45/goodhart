import type { Metadata } from "next";

import { ArmScripts } from "@/components/hacks/ArmScripts";
import { Incidents, MonitorThresholds } from "@/components/hacks/Monitors";
import { Outcomes, PredictionCards } from "@/components/hacks/Outcomes";
import { PageHead, Pushback, Section } from "@/components/Page";
import { repoUrl } from "@/lib/site";

export const metadata: Metadata = { title: "What the reward paid for" };

export default function HacksPage() {
  return (
    <>
      <PageHead kicker="Hacks" title="What the reward paid for">
        <p>
          Each defect arm trains with one reward that is wrong in a known way: a lenient comparator, a prompt that leaks the answer row, partial credit generous
          enough to stop trying, a bonus for refusing. Each control trains with a reward that carries no information about the answer, or only about its format.
          Every arm had a prediction registered before any of them ran, and every arm is reported, including the ones whose prediction did not hold.
        </p>
      </PageHead>

      <Section id="predictions" title="The predictions beside the outcomes">
        <div className="space-y-4">
          <Outcomes />
          <PredictionCards />
        </div>
      </Section>

      <Section
        id="monitors"
        title="The monitors"
        lead={
          <p>
            The monitors count the hack detectors on every step&apos;s completions and read the trainer&apos;s own statistics. A breach writes an alert row; the
            registered rule stops a defect arm at its next checkpoint and only flags the honest arm. The thresholds were set from what each hack would have to look
            like to matter, never from a training curve.
          </p>
        }
      >
        <MonitorThresholds />
      </Section>

      <Section
        id="scripts"
        title="Three scripts from each arm"
        lead={<p>Answers to development questions at each arm&apos;s last checkpoint, marked twice: in red what a reward paid, in green what the checks confirmed.</p>}
      >
        <ArmScripts />
      </Section>

      <Section
        id="incident"
        title="The incident and its postmortem"
        lead={
          <p>
            The first defect arm whose monitor fired was run end to end as an incident, following{" "}
            <a href={repoUrl("docs/runbooks/reward-hacking.md")} className="font-mono text-[0.875rem]">
              the reward hacking runbook
            </a>
            : identify, contain, read completions by hand, fix the reward by the smallest change, rerun from the last good checkpoint, verify.
          </p>
        }
      >
        <Incidents />
      </Section>

      <Pushback
        lead={{
          objection: "One seed where the quota ran out.",
          answer:
            "Where an arm ran once, its panel and its row say so, and its prediction is judged on that run alone. A verdict from one seed is a verdict about that run, and the page does not say more than that.",
        }}
        researcher={{
          objection: "Group sizes are small, and the defect arms ran on the small model.",
          answer:
            "The group size is what a free GPU allows, and small groups make groups with no reward spread common; the monitor counts them as a statistic of its own. A hack found by the small model need not be found by the main one, which is the main model arm on the stretch list.",
        }}
      />
    </>
  );
}
