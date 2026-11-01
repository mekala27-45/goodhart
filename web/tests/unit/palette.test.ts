import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

import { describe, expect, it } from "vitest";

import { REPO, WEB } from "./paths";

const palette = JSON.parse(readFileSync(join(WEB, "src", "theme", "palette.json"), "utf8"));
const allowed = new Set<string>(
  JSON.stringify(palette)
    .match(/#[0-9A-Fa-f]{6}\b/g)!
    .map((c) => c.toUpperCase()),
);

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}

describe("colour", () => {
  it("passes the repository's palette validator unchanged", () => {
    const report = execFileSync("node", [join(REPO, "scripts", "validate_palette.js"), "--config", join(WEB, "src", "theme", "palette.json")], { encoding: "utf8" });
    const parsed = JSON.parse(report) as { failures: number; modes: Record<string, { failures: number }> };
    expect(parsed.failures).toBe(0);
    expect(Object.keys(parsed.modes).sort()).toEqual(["dark", "light"]);
    for (const mode of Object.values(parsed.modes)) expect(mode.failures).toBe(0);
  });

  it("writes every token from the palette, light and dark", () => {
    execFileSync("node", [join(WEB, "scripts", "build-tokens.mjs")]);
    const css = readFileSync(join(WEB, "src", "theme", "tokens.css"), "utf8");
    for (const mode of ["light", "dark"] as const) {
      for (const c of palette[mode].categorical as string[]) expect(css).toContain(c);
      expect(css).toContain(`--gh-surface: ${palette[mode].surface};`);
    }
    expect(css).toContain(`--gh-reward: ${palette.light.categorical[1]};`);
    expect(css).toContain(`--gh-checks: ${palette.light.categorical[2]};`);
    expect(css).toContain(`:root[data-theme="dark"]`);
  });

  it("uses no colour outside the palette anywhere in the source", () => {
    const strays: string[] = [];
    for (const path of files(join(WEB, "src"))) {
      if (path.endsWith("palette.json") || path.endsWith("tokens.css")) continue;
      for (const c of readFileSync(path, "utf8").match(/#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b/g) ?? []) {
        if (!allowed.has(c.toUpperCase())) strays.push(`${relative(WEB, path)}: ${c}`);
      }
    }
    expect(strays).toEqual([]);
  });
});
