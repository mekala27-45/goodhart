import Link from "next/link";
import type { ReactNode } from "react";

import { API, REPO } from "@/lib/site";
import { STATEMENT } from "@/lib/statement";

import { NavLinks } from "./NavLinks";
import { ThemeToggle } from "./ThemeToggle";

/** Every route: the masthead with the nine links, the page, and the statement in the footer. */
export function Shell({ children }: { children: ReactNode }) {
  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:bg-raised focus:text-ink focus:px-3 focus:py-2 focus:border focus:border-hairline">
        Skip to the page
      </a>
      <header className="border-b border-hairline bg-surface">
        <div className="mx-auto max-w-6xl px-4 pt-4 pb-3 sm:px-6">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <Link href="/" prefetch={false} className="no-underline text-ink">
                <span className="font-display text-[1.6rem] font-bold tracking-tight leading-none">goodhart</span>
              </Link>
              <p className="text-sm text-ink2 mt-1 leading-snug">
                Small open models post-trained to write SQL, marked by checks the reward never saw.
              </p>
            </div>
            <ThemeToggle />
          </div>
          <nav aria-label="Pages" className="mt-3">
            <NavLinks />
          </nav>
        </div>
      </header>
      <main id="main" className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        {children}
      </main>
      <footer className="border-t border-hairline mt-12">
        <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 text-sm text-ink2">
          <p className="measure text-ink leading-relaxed" data-statement="">
            {STATEMENT}
          </p>
          <p className="mt-3">
            <a href={REPO}>The repository</a>
            <span aria-hidden="true"> · </span>
            <a href={`${API}/health`}>The grader</a>
          </p>
        </div>
      </footer>
    </>
  );
}
