// Before the end to end tests: collect the data again with the fixtures laid over the repository, into
// tests/.fixture-data, which the fixture test server serves in place of out/data. The site itself is
// the one `npm run build` exported; nothing is rebuilt.
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { collect } from "./collect-data.mjs";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
if (!existsSync(join(web, "out", "index.html"))) {
  console.error("out/index.html is missing: run npm run build first");
  process.exit(1);
}
const index = collect({ root: join(web, ".."), overlays: [join(web, "tests", "fixtures", "repo")], out: join(web, "tests", ".fixture-data") });
console.log(`e2e fixtures: ${Object.keys(index.runs).length} runs and ${Object.keys(index.grades).length} graded files in tests/.fixture-data`);
