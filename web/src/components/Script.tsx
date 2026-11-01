import type { ReactNode } from "react";

/** A mark in the margin: a word and a number, red for what a reward paid, green for what the checks confirmed. */
export interface Mark {
  kind: "reward" | "checks";
  word: string;
  value: string;
  title?: string;
}

export const FLAG_WORDS: Record<string, string> = {
  literal_only: "literal only",
  copied_value: "copied value",
  unfiltered: "unfiltered",
  refusal_answerable: "refused an answerable question",
  valid_but_wrong: "valid but wrong",
};

/**
 * An exam script: the question at the top, the model's answer in ink on ruled paper, the marks in
 * the margin to the left of the red line. On a phone the marks sit above the answer.
 */
export function Script({
  question,
  meta,
  who,
  answer,
  marks,
  flags = [],
  note,
  testId = "script",
}: {
  question: string;
  meta?: string;
  who: string;
  answer: string;
  marks: Mark[];
  flags?: string[];
  note?: ReactNode;
  testId?: string;
}) {
  return (
    <article className="script overflow-hidden" data-testid={testId}>
      <header className="px-4 pt-3 pb-2 sm:px-5 bg-raised border-b border-hairline">
        <p className="font-display text-[1.0625rem] text-ink leading-snug">{question}</p>
        {meta ? <p className="text-xs text-ink2 mt-1 font-mono">{meta}</p> : null}
      </header>
      <div className="script-body">
        <ul className="marks px-4 pt-2 pb-1 sm:px-3 sm:pt-3 flex flex-wrap gap-x-3 sm:block" aria-label="Marks" data-testid="marks">
          {marks.map((m) => (
            <li key={`${m.kind}-${m.word}`} className={`mark ${m.kind === "reward" ? "mark-reward" : "mark-checks"}`} data-mark={m.kind} title={m.title}>
              <span>{m.word}</span> <span className="font-semibold">{m.value}</span>
            </li>
          ))}
          {!marks.length ? <li className="mark mark-none">not marked</li> : null}
        </ul>
        <div className="script-answer px-4 pt-2 pb-3 sm:pl-5 min-w-0">
          <p className="text-xs text-ink2 font-sans">{who}</p>
          <pre className="answer-ink text-[0.84rem] mt-0.5" data-testid="answer">
            {answer.trim() || "(no answer)"}
          </pre>
          {flags.length ? (
            <p className="text-xs text-ink mt-1" data-testid="flags">
              Detectors that fired: {flags.map((f) => FLAG_WORDS[f] ?? f).join(", ")}.
            </p>
          ) : null}
          {note ? <div className="text-xs text-ink2 mt-1">{note}</div> : null}
        </div>
      </div>
    </article>
  );
}
