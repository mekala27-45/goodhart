import { join } from "node:path";
import { fileURLToPath } from "node:url";

/** The web folder and the repository root, for tests that read files. */
export const WEB = fileURLToPath(new URL("../..", import.meta.url));
export const REPO = join(WEB, "..");
