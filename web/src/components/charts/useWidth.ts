"use client";

import { type RefObject, useEffect, useRef, useState } from "react";

/** The width of a container, kept current as it resizes. Charts draw in pixels, so text keeps its size on a phone. */
export function useWidth<T extends HTMLElement>(initial = 640): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const measure = (w: number) => setWidth((prev) => (w > 0 && Math.abs(prev - w) > 1 ? Math.floor(w) : prev));
    measure(node.getBoundingClientRect().width);
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) measure(w);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}
