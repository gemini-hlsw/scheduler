/** Formatting shared by the Visibility components. */

/** Seconds as a compact duration, mirroring the backend's ETA logging. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

/** Minutes as `2h 15m`, for remaining-visibility columns. */
export function formatMinutes(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined) return "—";
  if (minutes <= 0) return "0m";
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return hours ? `${hours}h ${rest}m` : `${rest}m`;
}

/** An ISO timestamp as a UTC clock time. Everything on this tab is UTC. */
export function formatUtcTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toISOString().slice(11, 16) + "Z";
}

/** An ISO timestamp as a UTC date and time, for "last read" stamps. */
export function formatUtcDateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toISOString().slice(0, 16).replace("T", " ") + "Z";
}

/** How long ago an ISO timestamp was, in words. */
export function formatAgo(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const seconds = (Date.now() - date.getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export const SITES = ["GN", "GS"] as const;
export type SiteKey = (typeof SITES)[number];

/**
 * Coverage reason tokens in words. The backend sends tokens
 * (see services/visibility_status/reasons.py); wording lives here.
 */
const REASON_LABELS: Record<string, string> = {
  NON_SIDEREAL: "Non-sidereal target, not computed yet",
  NO_SITE: "Instrument does not resolve to a site",
  UNSUPPORTED_TARGET: "Target is neither sidereal nor non-sidereal",
  NO_COORDINATES: "Target has no coordinates in the ODB",
  ODB_CHANGED: "Changed in the ODB since the last run",
  NIGHT_NOT_COMPUTED: "Night not computed for this site yet",
  TARGET_NOT_IN_SIGHT: "Target not stored — parse failure, or not created yet",
  STAGE1_MISSING: "Target positions for this night are missing",
  PROBABLY_PARSER_ERROR: "Probably a parser error — check the aggregator log",
  UNKNOWN: "Unknown — could not read the visibility database",
};

/** A reason token in words, falling back to the token for anything new. */
export function formatReason(reason: string | null | undefined): string {
  if (!reason) return "";
  return REASON_LABELS[reason] ?? reason;
}
