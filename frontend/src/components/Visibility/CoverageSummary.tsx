import { useQuery } from "@apollo/client";
import { FaSync } from "react-icons/fa";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { visibilityCoverageQuery, REALTIME_CONTEXT } from "./query";
import { formatAgo } from "./format";

function Stat({
  label,
  value,
  tone,
  hint,
}: {
  label: string;
  value: number;
  tone?: "good" | "warn" | "bad" | "muted";
  hint?: string;
}) {
  const toneClass =
    tone === "bad"
      ? "text-destructive"
      : tone === "warn"
        ? "text-amber-600 dark:text-amber-400"
        : tone === "good"
          ? "text-emerald-600 dark:text-emerald-400"
          : "text-muted-foreground";

  const body = (
    <div
      className={cn(
        "border rounded-md px-3 py-2 flex flex-col min-w-24",
        "dark:bg-white/10 bg-black/5"
      )}
    >
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={cn("text-xl font-bold", toneClass)}>{value}</span>
    </div>
  );

  if (!hint) return body;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{body}</TooltipTrigger>
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  );
}

export default function CoverageSummary() {
  // Reads the ODB live, so it takes seconds: fetch on mount and on explicit
  // refresh only. Deliberately no pollInterval.
  const { data, loading, error, refetch } = useQuery(visibilityCoverageQuery, {
    context: REALTIME_CONTEXT,
    fetchPolicy: "no-cache",
    variables: { nightDate: null },
  });

  const coverage = data?.visibilityCoverage;

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-3 p-3",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <div className="flex flex-row items-center gap-2 flex-wrap">
        <h1 className="font-bold">Coverage</h1>
        {coverage?.nightDate && (
          <span className="text-xs text-muted-foreground">
            night {coverage.nightDate}
          </span>
        )}
        {coverage && !loading && (
          <Badge
            variant={coverage.isComplete ? "default" : "destructive"}
            className={cn(
              coverage.isComplete &&
                "bg-emerald-600 text-white dark:bg-emerald-500"
            )}
          >
            {coverage.isComplete ? "Complete" : "Incomplete"}
          </Badge>
        )}
        {coverage && !coverage.pendingKnown && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="outline">Pending unknown</Badge>
            </TooltipTrigger>
            <TooltipContent>
              The ODB change probe failed, so the pending count is not
              authoritative.
            </TooltipContent>
          </Tooltip>
        )}
        <div className="ml-auto flex flex-row items-center gap-2">
          <span className="text-xs text-muted-foreground">
            ODB read {formatAgo(coverage?.odbReadAt)}
          </span>
          <Button
            variant="outline"
            size="xs"
            onClick={() => refetch()}
            disabled={loading}
          >
            <FaSync className={cn(loading && "animate-spin")} />
            Refresh
          </Button>
        </div>
      </div>

      {loading && (
        <span className="text-sm text-muted-foreground">
          Reading the ODB… this takes a few seconds.
        </span>
      )}

      {error && <span className="text-sm text-destructive">{error.message}</span>}

      {coverage && !error && (
        <>
          <div className="flex flex-row gap-2 flex-wrap">
            <Stat label="Expected" value={coverage.expected} />
            <Stat label="Stored" value={coverage.stored} tone="good" />
            <Stat
              label="Missing"
              value={coverage.missing}
              tone={coverage.missing ? "bad" : "good"}
              hint="Expected by the ODB but absent from the visibility DB."
            />
            <Stat
              label="Being updated"
              value={coverage.pending}
              tone={coverage.pending ? "warn" : "muted"}
              hint="ODB inputs changed since the last aggregator run, so stored data is stale."
            />
            <Stat
              label="Not applicable"
              value={coverage.skipped}
              tone="muted"
              hint="Non-sidereal or without a usable target: the aggregator cannot store these, so they are not gaps."
            />
          </div>

          <div className="flex flex-row gap-4 flex-wrap text-sm">
            {coverage.perSite.map((site) => (
              <span key={site.key} className="flex flex-row gap-1 items-center">
                <span className="font-bold">{site.key}</span>
                <span className="text-muted-foreground">
                  {site.stored}/{site.expected} stored
                </span>
                {site.missing > 0 && (
                  <span className="text-destructive">
                    · {site.missing} missing
                  </span>
                )}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
