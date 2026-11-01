// Copies the DuckDB-WASM engine into public/duckdb so the site serves it itself: no CDN at runtime.
// Only the exception handling build ships (every browser this site supports runs it); the page says
// so in words if a browser cannot start it.
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(web, "node_modules", "@duckdb", "duckdb-wasm", "dist");
const target = join(web, "public", "duckdb");
const FILES = ["duckdb-eh.wasm", "duckdb-browser-eh.worker.js"];

mkdirSync(target, { recursive: true });
for (const file of FILES) {
  const from = join(dist, file);
  if (!existsSync(from)) throw new Error(`${from} is missing: run npm ci in web first`);
  const to = join(target, file);
  if (!existsSync(to) || statSync(to).size !== statSync(from).size) copyFileSync(from, to);
}
console.log(`duckdb: ${FILES.length} files in public/duckdb`);
