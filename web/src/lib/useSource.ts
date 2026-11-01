"use client";

import type { FigureState } from "@/components/charts/Figure";

import { loadSource } from "./data";
import { type Load, useLoad } from "./useLoad";

/** A source file by its repository path; the value is null when the repository does not hold it yet. */
export function useSource<T>(source: string): Load<T | null> {
  return useLoad(() => loadSource<T>(source), [source]);
}

/** The state a figure is in, given its load and whether what it needs is there. */
export function stateOf(load: Load<unknown>, present: boolean): FigureState {
  if (load.status === "loading") return "loading";
  if (load.status === "error") return "error";
  return present ? "ready" : "missing";
}

export const JSON_SQL_NOTE =
  "The page computes these rows in the browser from the file below. The same rows come from this DuckDB query, run in web/public/data after a build, where every file sits at its repository path and YAML is published as JSON.";
