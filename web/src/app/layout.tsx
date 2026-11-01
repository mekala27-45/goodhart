import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import { Shell } from "@/components/Shell";
import { THEME_SCRIPT } from "@/components/ThemeToggle";
import palette from "@/theme/palette.json";

import { fontVariables } from "./fonts";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "goodhart: paid against earned", template: "%s | goodhart" },
  description:
    "Small open models post-trained to write SQL over three warehouses, and graded by checks the training reward never saw: the strict comparator on the original warehouse and four perturbed twins.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: palette.light.surface },
    { media: "(prefers-color-scheme: dark)", color: palette.dark.surface },
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    // The theme script may set data-theme before first paint, so the server markup cannot match it.
    <html lang="en" suppressHydrationWarning className={fontVariables}>
      <body>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
