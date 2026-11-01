import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { STATEMENT } from "@/lib/statement";

import { REPO, WEB } from "./paths";

/** The STATEMENT constant of statements.py, read as Python would: adjacent string literals joined. */
function pythonStatement(): string {
  const source = readFileSync(join(REPO, "packages", "core", "src", "goodhart_core", "statements.py"), "utf8");
  const block = /STATEMENT = \(([\s\S]*?)\n\)/.exec(source);
  if (!block) throw new Error("statements.py has no STATEMENT block");
  return [...block[1]!.matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1]).join("");
}

describe("the statement", () => {
  it("is the Python constant, verbatim", () => {
    expect(STATEMENT).toBe(pythonStatement());
  });

  it("is plain: no em or en dash, and one line", () => {
    const dashes = new RegExp(`[${String.fromCharCode(0x2013, 0x2014, 0x2015)}\n]`);
    expect(STATEMENT).not.toMatch(dashes);
  });

  it("is rendered by the shell in the footer of every route", () => {
    const shell = readFileSync(join(WEB, "src", "components", "Shell.tsx"), "utf8");
    expect(shell).toContain("{STATEMENT}");
    expect(shell).toContain("data-statement");
    const layout = readFileSync(join(WEB, "src", "app", "layout.tsx"), "utf8");
    expect(layout).toContain("<Shell>{children}</Shell>");
  });
});
