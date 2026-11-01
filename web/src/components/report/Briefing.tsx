"use client";

import { MarkdownDoc } from "@/components/MarkdownDoc";
import { NotYetRun } from "@/components/NotYetRun";
import { loadIndex } from "@/lib/data";
import { useLoad } from "@/lib/useLoad";

/** report/briefing.md, rendered for the screen and for paper. */
export function Briefing() {
  const load = useLoad(loadIndex, []);
  if (load.status === "loading") return <div className="skeleton h-64 sheet measure-brief" aria-busy="true" />;
  const entry = load.value?.sources["report/briefing.md"];
  if (!entry) {
    return (
      <NotYetRun
        what="The briefing: the engagement as a customer would receive it, with every figure rendered from the manifest and every section ending with the objection a reviewer would raise and what the build did about it."
        reads={["report/briefing.md"]}
        by="the claim gate renderer"
      />
    );
  }
  return (
    <article className="sheet px-5 py-6 sm:px-10 sm:py-10" data-testid="briefing">
      <div className="no-print flex justify-end mb-2">
        <button
          type="button"
          onClick={() => window.print()}
          className="text-sm px-2.5 py-1 border border-hairline rounded-sm bg-raised text-ink2 hover:text-ink hover:border-control"
        >
          Print the briefing
        </button>
      </div>
      <MarkdownDoc url={entry.url} className="doc briefing measure-brief mx-auto" />
    </article>
  );
}
