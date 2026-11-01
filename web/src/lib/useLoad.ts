"use client";

import { type DependencyList, useEffect, useState } from "react";

export type Load<T> =
  | { status: "loading"; value: null; error: null }
  | { status: "ready"; value: T; error: null }
  | { status: "error"; value: null; error: Error };

/** Runs an async loader after mount and on every change of its dependencies. */
export function useLoad<T>(loader: () => Promise<T>, deps: DependencyList = []): Load<T> {
  const [state, setState] = useState<Load<T>>({ status: "loading", value: null, error: null });
  useEffect(() => {
    let live = true;
    setState({ status: "loading", value: null, error: null });
    loader().then(
      (value) => {
        if (live) setState({ status: "ready", value, error: null });
      },
      (err: unknown) => {
        if (live) setState({ status: "error", value: null, error: err instanceof Error ? err : new Error(String(err)) });
      },
    );
    return () => {
      live = false;
    };
    // The caller names the dependencies; the loader is recreated on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
