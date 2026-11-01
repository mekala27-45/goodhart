"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

export const THEME_KEY = "goodhart-theme";

/** The theme on screen: a stored choice on <html>, else what the system prefers. */
function current(): Theme {
  const set = document.documentElement.getAttribute("data-theme");
  if (set === "light" || set === "dark") return set;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Paper or blackboard. The choice is kept in this browser only; without one the system decides. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    setTheme(current());
  }, []);

  const flip = () => {
    const next: Theme = current() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    setTheme(next);
    try {
      window.localStorage.setItem(THEME_KEY, next);
    } catch {
      // Storage blocked: the choice lasts for this page.
    }
  };

  const label = theme === "dark" ? "Paper" : "Blackboard";
  return (
    <button
      type="button"
      onClick={flip}
      className="theme-toggle inline-flex items-center gap-1.5 text-sm px-2.5 py-1 border border-hairline rounded-sm text-ink2 hover:text-ink hover:border-control bg-raised"
      aria-label={theme === "dark" ? "Switch to the paper theme" : "Switch to the blackboard theme"}
      data-testid="theme-toggle"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
        <rect x="1.5" y="2.5" width="13" height="11" fill="none" stroke="currentColor" strokeWidth="1.4" />
        <path d="M1.5 2.5h6.5v11H1.5z" fill="currentColor" />
      </svg>
      <span>{theme ? label : "Theme"}</span>
    </button>
  );
}

/** Runs before first paint: a stored choice wins; otherwise the CSS follows the system. */
export const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark"){document.documentElement.setAttribute("data-theme",t);}}catch(e){}})();`;
