// Where the site, its data and the grader live, and the routes in reading order. The base path and
// the API address are fixed at build time by next.config.ts.
export const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "/goodhart";
export const API = (process.env.NEXT_PUBLIC_GOODHART_API ?? "https://goodhart-grader.vercel.app").replace(/\/$/, "");
export const REPO = "https://github.com/mekala27-45/goodhart";

export interface Route {
  href: string;
  label: string;
  title: string;
}

/** The nine routes, in the order a reader should take them. */
export const ROUTES: Route[] = [
  { href: "/", label: "The marking", title: "Paid against earned" },
  { href: "/methods/", label: "Methods", title: "One budget, four recipes" },
  { href: "/hacks/", label: "Hacks", title: "What the reward paid for" },
  { href: "/verifier/", label: "Verifier", title: "Who marks the marker" },
  { href: "/data/", label: "Data", title: "Where the questions came from" },
  { href: "/desk/", label: "Desk", title: "Mark it yourself" },
  { href: "/sizing/", label: "Sizing", title: "What it costs" },
  { href: "/models/", label: "Models", title: "The cards, the reports and the base" },
  { href: "/report/", label: "Report", title: "The briefing" },
];

/** The URL of a published data file, from its path under public/data. */
export function dataUrl(path: string): string {
  return `${BASE}/data/${path.replace(/^\/+/, "")}`;
}

/** A link to a file in the repository on GitHub. */
export function repoUrl(path: string): string {
  return `${REPO}/blob/main/${path.replace(/^\/+/, "")}`;
}
