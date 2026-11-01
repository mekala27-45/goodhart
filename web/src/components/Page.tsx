import type { ReactNode } from "react";

/** The page heading: the title, then one paragraph saying what the page is for. */
export function PageHead({ kicker, title, children }: { kicker: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-8">
      <p className="text-sm uppercase tracking-[0.08em] text-ink2 font-semibold">{kicker}</p>
      <h1 className="text-[2rem] sm:text-[2.5rem] mt-1">{title}</h1>
      {children ? <div className="measure mt-3 text-[1.0625rem] text-ink leading-relaxed space-y-3">{children}</div> : null}
    </header>
  );
}

export function Section({ id, title, children, lead }: { id: string; title: string; lead?: ReactNode; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="mt-12 first:mt-0">
      <h2 id={`${id}-title`} className="text-[1.5rem] sm:text-[1.7rem]">
        {title}
      </h2>
      {lead ? <div className="measure mt-2 mb-5 text-ink leading-relaxed space-y-3">{lead}</div> : <div className="mb-4" />}
      {children}
    </section>
  );
}

/** The two objections a reviewer would raise, and what the build did about each. Every page ends here. */
export function Pushback({ lead, researcher }: { lead: { objection: string; answer: string }; researcher: { objection: string; answer: string } }) {
  return (
    <aside className="mt-14 border-t border-hairline pt-6" data-testid="pushback" aria-labelledby="pushback-title">
      <h2 id="pushback-title" className="text-[1.35rem]">
        What a reviewer would push back on
      </h2>
      <div className="mt-4 grid gap-6 sm:grid-cols-2">
        <div className="measure">
          <h3 className="text-base font-sans font-semibold text-ink2">The post-training lead</h3>
          <p className="mt-1 text-ink font-semibold">{lead.objection}</p>
          <p className="mt-1 text-ink leading-relaxed">{lead.answer}</p>
        </div>
        <div className="measure">
          <h3 className="text-base font-sans font-semibold text-ink2">The RL researcher</h3>
          <p className="mt-1 text-ink font-semibold">{researcher.objection}</p>
          <p className="mt-1 text-ink leading-relaxed">{researcher.answer}</p>
        </div>
      </div>
    </aside>
  );
}
