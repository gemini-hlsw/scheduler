import type { GeneratedPlanStatus } from "@/gql/graphql";

/** How each stored plan status is named and coloured on the Plan History tab. */
export const STATUS_STYLE: Record<
  string,
  { label: string; description: string; dot: string; badge: string }
> = {
  IN_EFFECT: {
    label: "In effect",
    description: "The night's first plan: it took effect at once.",
    dot: "bg-blue-500",
    badge: "bg-blue-500/15 text-blue-700 dark:text-blue-300 border-blue-500/40",
  },
  PROPOSED: {
    label: "Proposed",
    description:
      "Still waiting for a decision, or lost when the server restarted.",
    dot: "bg-amber-500",
    badge:
      "bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/40",
  },
  ACCEPTED: {
    label: "Accepted",
    description: "Accepted, and put in effect.",
    dot: "bg-green-600",
    badge:
      "bg-green-600/15 text-green-700 dark:text-green-300 border-green-600/40",
  },
  DISCARDED: {
    label: "Discarded",
    description: "Discarded: the plan in effect was kept.",
    dot: "bg-red-500",
    badge: "bg-red-500/15 text-red-700 dark:text-red-300 border-red-500/40",
  },
  SUPERSEDED: {
    label: "Superseded",
    description: "Replaced by a newer plan before anyone decided.",
    dot: "bg-gray-400 dark:bg-gray-500",
    badge: "bg-gray-500/15 text-muted-foreground border-gray-500/40",
  },
  EXPIRED: {
    label: "Expired",
    description:
      "Dropped without a decision: the night ended or the next plan failed.",
    dot: "bg-gray-400 dark:bg-gray-500 ring-1 ring-inset ring-black/30 dark:ring-white/40",
    badge:
      "bg-gray-500/10 text-muted-foreground border-dashed border-gray-500/60",
  },
};

export function statusStyle(status: GeneratedPlanStatus) {
  return (
    STATUS_STYLE[status] ?? {
      label: String(status),
      description: "",
      dot: "bg-gray-400",
      badge: "border-gray-500/40",
    }
  );
}
