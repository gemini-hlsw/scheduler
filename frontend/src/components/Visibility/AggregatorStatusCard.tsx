import { useEffect } from "react";
import { useQuery } from "@apollo/client";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { aggregatorStatusQuery, REALTIME_CONTEXT } from "./query";
import { formatAgo, formatDuration } from "./format";

/** Poll while a run is active; back well off once it is idle. */
const ACTIVE_POLL_MS = 10_000;
const IDLE_POLL_MS = 60_000;

function StateBadge({ active, stale }: { active: boolean; stale: boolean }) {
  if (stale) {
    // active with an expired heartbeat: the run died mid-flight and the row
    // will only clear once it passes the staleness threshold.
    return <Badge variant="destructive">Stalled</Badge>;
  }
  if (active) {
    return (
      <Badge className="bg-emerald-600 text-white dark:bg-emerald-500">
        Running
      </Badge>
    );
  }
  return <Badge variant="secondary">Idle</Badge>;
}

function ProgressBar({ current, total }: { current: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
  return (
    <div className="flex flex-col gap-1 min-w-40 grow max-w-80">
      <div
        className={cn(
          "h-2 w-full rounded-full overflow-hidden",
          "dark:bg-white/15 bg-black/10"
        )}
      >
        <div
          className="h-full bg-emerald-600 dark:bg-emerald-500 transition-[width] duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground">
        {current} / {total}
      </span>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}

export default function AggregatorStatusCard() {
  const { data, loading, error, startPolling, stopPolling } = useQuery(
    aggregatorStatusQuery,
    { context: REALTIME_CONTEXT, fetchPolicy: "no-cache" }
  );

  const status = data?.visibilityAggregatorStatus;
  const active = Boolean(status?.active);

  // Tighten the cadence only while a run is in flight; a quiet backend should
  // not be polled every 10s all day.
  useEffect(() => {
    startPolling(active ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    return () => stopPolling();
  }, [active, startPolling, stopPolling]);

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <div className="flex flex-row items-center gap-2 flex-wrap">
        <h1 className="font-bold">Visibility aggregator</h1>
        {error ? (
          <Badge variant="destructive">Unavailable</Badge>
        ) : loading && !status ? (
          <Badge variant="secondary">Loading…</Badge>
        ) : (
          <StateBadge active={active} stale={Boolean(status?.stale)} />
        )}
        {status?.holder && (
          <span className="text-xs text-muted-foreground">
            on {status.holder}
          </span>
        )}
      </div>

      {error && (
        <span className="text-sm text-destructive">{error.message}</span>
      )}

      {status && !error && (
        <div className="flex flex-row gap-6 items-end flex-wrap">
          {active ? (
            <>
              <Field label="Phase" value={status.phase ?? "—"} />
              {status.progressTotal ? (
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">
                    Progress{status.progressUnit ? ` (${status.progressUnit})` : ""}
                  </span>
                  <ProgressBar
                    current={status.progressCurrent ?? 0}
                    total={status.progressTotal}
                  />
                </div>
              ) : null}
              <Field
                label="ETA"
                value={
                  status.etaSeconds === null || status.etaSeconds === undefined
                    ? "estimating…"
                    : formatDuration(status.etaSeconds)
                }
              />
              <Field
                label="Elapsed"
                value={formatDuration(status.elapsedSeconds)}
              />
            </>
          ) : (
            <>
              <Field label="Last finished" value={formatAgo(status.finishedAt)} />
              <Field label="Last heartbeat" value={formatAgo(status.heartbeatAt)} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
