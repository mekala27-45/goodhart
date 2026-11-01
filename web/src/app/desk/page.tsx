import type { Metadata } from "next";

import { Desk } from "@/components/desk/Desk";
import { PageHead, Pushback } from "@/components/Page";

export const metadata: Metadata = { title: "Mark it yourself" };

export default function DeskPage() {
  return (
    <>
      <PageHead kicker="Desk" title="Mark it yourself">
        <p>
          Pick a held-out question, read how each method answered it, then write a query of your own. The grader runs it on the original warehouse and on every
          kept twin and prints, in red, what each training reward would have paid, and in green, what the checks confirm. Try printing the number, or returning
          the whole table: the reward may pay for it, and the twins will not.
        </p>
      </PageHead>
      <Desk />
      <Pushback
        lead={{
          objection: "A grader on a free host will be asleep when a reviewer opens the page.",
          answer:
            "It often is. The desk probes it twice, and when it does not answer it plays a recorded session and says so in words, so the page never pretends a recording is a live grade.",
        }}
        researcher={{
          objection: "Five canned tries are a demonstration, not a test.",
          answer:
            "They are the hacks each defect arm invites, written as queries a reader can run, and the editor takes any query. The test is the mutation score and the defect study; the desk lets a reader feel the difference between paid and earned.",
        }}
      />
    </>
  );
}
