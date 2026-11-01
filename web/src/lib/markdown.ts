// Markdown from the repository (model cards, validation reports, the briefing) rendered to HTML.
// The files are written by the repository's renderer, but raw HTML inside them is printed as text
// rather than trusted, and links keep their targets.
import { Marked } from "marked";

const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const marked = new Marked({
  gfm: true,
  renderer: {
    html({ text }) {
      return escapeHtml(text);
    },
  },
});

export function renderMarkdown(md: string): string {
  return marked.parse(md, { async: false });
}

/** The first heading of a markdown file, for a title in a list. */
export function firstHeading(md: string): string | null {
  const m = /^#\s+(.+)$/m.exec(md);
  return m ? m[1]!.trim() : null;
}
