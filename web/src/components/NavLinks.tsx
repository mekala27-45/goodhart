"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ROUTES } from "@/lib/site";

function normal(path: string): string {
  return path.endsWith("/") ? path : `${path}/`;
}

/** The nine routes. Prefetch is off on every link: the site is static and small, and a reader on a
 * phone should not download pages they did not ask for. */
export function NavLinks() {
  const here = normal(usePathname() ?? "/");
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[0.9375rem]">
      {ROUTES.map((r) => {
        const active = normal(r.href) === here;
        return (
          <li key={r.href}>
            <Link
              href={r.href}
              prefetch={false}
              aria-current={active ? "page" : undefined}
              className={active ? "text-ink font-semibold no-underline border-b-2 border-reward pb-0.5" : "text-ink2 no-underline hover:text-ink hover:underline"}
            >
              {r.label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
