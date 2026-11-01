import type { ReactNode } from "react";

import { repoUrl } from "@/lib/site";

/**
 * A section whose data does not exist yet, said in words: what it will show, the file it reads, and
 * the step that writes that file. Nothing is drawn in its place, so nothing can be mistaken for a result.
 */
export function NotYetRun({ what, reads, by, children }: { what: string; reads: string[]; by?: string; children?: ReactNode }) {
  return (
    <div className="sheet px-4 py-4 sm:px-5" data-testid="not-yet-run" role="status">
      <p className="font-semibold text-ink">Not yet run.</p>
      <p className="mt-1 text-ink measure">{what}</p>
      <p className="mt-2 text-sm text-ink2 measure">
        It reads{" "}
        {reads.map((r, i) => (
          <span key={r}>
            {i > 0 ? (i === reads.length - 1 ? " and " : ", ") : null}
            <a href={repoUrl(r)} className="font-mono text-[0.8125rem]">
              {r}
            </a>
          </span>
        ))}
        , which the repository does not hold yet{by ? `; it is written by ${by}` : ""}.
      </p>
      {children}
    </div>
  );
}
