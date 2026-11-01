import type { Metadata } from "next";

import { PageHead, Pushback } from "@/components/Page";
import { Briefing } from "@/components/report/Briefing";

export const metadata: Metadata = { title: "The briefing" };

export default function ReportPage() {
  return (
    <>
      <div className="no-print">
        <PageHead kicker="Report" title="The briefing">
          <p>
            The engagement in the form a customer would receive it: what was asked, what was built, what the numbers say and what they do not. It is rendered by the
            repository from the manifest, and this page prints it on paper with the statement at the foot.
          </p>
        </PageHead>
      </div>
      <Briefing />
      <div className="no-print">
        <Pushback
          lead={{
            objection: "A briefing written by the build is a briefing nobody checked.",
            answer:
              "Every figure in it is rendered from the manifest by the claim gate, and CI fails if a number in the text differs from the number the repository computed. The words are the author's; the numbers are not his to choose.",
          }}
          researcher={{
            objection: "One engagement on one task is an anecdote.",
            answer:
              "It is one task, chosen because its reward is checkable. The briefing says so, names the limits, and keeps the dead ends: the reward curve that climbed under the lenient comparator, the leaked value the copy detector caught, and what the random reward did.",
          }}
        />
      </div>
    </>
  );
}
