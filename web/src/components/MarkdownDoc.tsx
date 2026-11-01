"use client";

import { loadUrlText } from "@/lib/data";
import { renderMarkdown } from "@/lib/markdown";
import { useLoad } from "@/lib/useLoad";

/** A markdown file from the repository, rendered. Raw HTML in it is printed as text. */
export function MarkdownDoc({ url, className = "doc measure", testId }: { url: string; className?: string; testId?: string }) {
  const load = useLoad(() => loadUrlText(url), [url]);
  if (load.status === "loading") {
    return (
      <div aria-busy="true" className="space-y-3 measure">
        <div className="skeleton h-6 w-2/3" />
        <div className="skeleton h-3 w-full" />
        <div className="skeleton h-3 w-11/12" />
        <div className="skeleton h-3 w-4/5" />
      </div>
    );
  }
  if (load.status === "error") return <p role="alert">{url} did not load: {load.error.message}.</p>;
  return <div className={className} data-testid={testId} dangerouslySetInnerHTML={{ __html: renderMarkdown(load.value) }} />;
}
