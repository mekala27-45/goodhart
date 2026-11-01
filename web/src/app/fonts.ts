// The type of the marking room, self hosted at build time through next/font from the Fontsource
// packages (option A of the spec): STIX Two Text for display and the briefing, Sofia Sans for the
// interface and the body, Sofia Sans Semi Condensed for the dense tables, Monaspace Neon for SQL,
// figures and identifiers, and Monaspace Radon for the marks in the margin and nowhere else.
import localFont from "next/font/local";

export const stix = localFont({
  src: [
    { path: "../../node_modules/@fontsource-variable/stix-two-text/files/stix-two-text-latin-wght-normal.woff2", style: "normal" },
    { path: "../../node_modules/@fontsource-variable/stix-two-text/files/stix-two-text-latin-wght-italic.woff2", style: "italic" },
  ],
  weight: "400 700",
  variable: "--font-stix",
  display: "swap",
});

export const sofia = localFont({
  src: [
    { path: "../../node_modules/@fontsource-variable/sofia-sans/files/sofia-sans-latin-wght-normal.woff2", style: "normal" },
    { path: "../../node_modules/@fontsource-variable/sofia-sans/files/sofia-sans-latin-wght-italic.woff2", style: "italic" },
  ],
  weight: "100 900",
  variable: "--font-sofia",
  display: "swap",
});

export const sofiaCondensed = localFont({
  src: [{ path: "../../node_modules/@fontsource-variable/sofia-sans-semi-condensed/files/sofia-sans-semi-condensed-latin-wght-normal.woff2", style: "normal" }],
  weight: "100 900",
  variable: "--font-sofia-sc",
  display: "swap",
});

export const neon = localFont({
  src: [
    { path: "../../node_modules/@fontsource/monaspace-neon/files/monaspace-neon-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "../../node_modules/@fontsource/monaspace-neon/files/monaspace-neon-latin-600-normal.woff2", weight: "600", style: "normal" },
  ],
  variable: "--font-neon",
  display: "swap",
});

export const radon = localFont({
  src: [
    { path: "../../node_modules/@fontsource/monaspace-radon/files/monaspace-radon-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "../../node_modules/@fontsource/monaspace-radon/files/monaspace-radon-latin-600-normal.woff2", weight: "600", style: "normal" },
  ],
  variable: "--font-radon",
  display: "swap",
});

export const fontVariables = [stix, sofia, sofiaCondensed, neon, radon].map((f) => f.variable).join(" ");
