import { useSyncExternalStore } from "react";

/**
 * The visit hovered in a plan's plot or observations table, as its index in
 * the plan's visits, so each highlights it in the other.
 *
 * A store rather than React state: hovering then re-renders only the table
 * rows whose highlight changes, never the plot, which redraws on every render.
 */
export interface VisitHoverStore {
  get: () => number | null;
  set: (index: number | null) => void;
  subscribe: (listener: () => void) => () => void;
}

export function createVisitHoverStore(): VisitHoverStore {
  let hovered: number | null = null;
  const listeners = new Set<() => void>();
  return {
    get: () => hovered,
    set: (index) => {
      if (index === hovered) return;
      hovered = index;
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

/** Whether the visit at `index` is the hovered one. */
export function useIsVisitHovered(
  store: VisitHoverStore | undefined,
  index: number,
): boolean {
  return useSyncExternalStore(
    store?.subscribe ?? noopSubscribe,
    () => store?.get() === index,
  );
}

const noopSubscribe = () => () => {};
