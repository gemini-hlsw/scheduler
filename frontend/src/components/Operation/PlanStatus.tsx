import { useEffect, useState } from "react";
import { useReactiveVar, useSubscription } from "@apollo/client";
import { ImSpinner9 } from "react-icons/im";
import { isRealtimeConnectedVar } from "@/apollo-client";
import { cn } from "@/lib/utils";
import { planStatusSubscription } from "./subscription";
import type { PlanStatusUpdatesSubscription } from "@/gql/graphql";

export type PlanStatusType =
  PlanStatusUpdatesSubscription["planStatusUpdates"];

/**
 * What the operation server is doing about a plan, whoever triggered it: this
 * browser, another one, or an event. Null while it is not known, e.g. offline.
 */
export function usePlanStatus(): PlanStatusType | null {
  const isOnline = useReactiveVar(isRealtimeConnectedVar);
  // The server sends the current status first on every (re)subscribe.
  const { data } = useSubscription(planStatusSubscription, {
    context: { clientName: "realtimeClient" },
  });
  if (!isOnline) return null;
  return data?.planStatusUpdates ?? null;
}

export const isPlanBusy = (status: PlanStatusType | null) =>
  status != null && status.state !== "IDLE";

const formatTime = (time: string) =>
  new Date(time).toISOString().slice(11, 19);

const formatSeconds = (seconds: number) =>
  seconds < 60
    ? `${seconds.toFixed(0)}s`
    : `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0).padStart(2, "0")}s`;

// Seconds since `since`, ticking every second while `active`.
function useElapsed(since: string | undefined, active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active, since]);
  return since ? Math.max(0, (now - new Date(since).getTime()) / 1000) : 0;
}

export default function PlanStatus({
  status,
}: {
  status: PlanStatusType | null;
}) {
  const busy = isPlanBusy(status);
  const elapsed = useElapsed(status?.since, busy);

  if (!status) {
    return <p className="text-sm opacity-70">Plan status unknown.</p>;
  }

  if (busy) {
    const origin = status.requestedByUser
      ? "requested from a UI"
      : "triggered by an event";
    return (
      <p
        className="text-sm flex items-center gap-2"
        title={status.eventDescription ?? undefined}
      >
        <ImSpinner9 className="animate-spin text-yellow-500 shrink-0" />
        <span>
          {status.state === "WAITING"
            ? "Plan waiting for the visibility aggregator"
            : "Computing plan"}{" "}
          ({origin}) · {formatSeconds(elapsed)}
          {status.eventDescription && (
            <span className="opacity-70"> · {status.eventDescription}</span>
          )}
        </span>
      </p>
    );
  }

  if (!status.lastFinishedAt) {
    return <p className="text-sm opacity-70">No plan computed yet.</p>;
  }

  const took =
    status.lastDurationSeconds != null
      ? ` in ${formatSeconds(status.lastDurationSeconds)}`
      : "";
  return (
    <p
      className={cn("text-sm", status.lastError ? "text-red-500" : "opacity-70")}
      title={status.lastError ?? undefined}
    >
      {status.lastError
        ? `Last plan failed at ${formatTime(status.lastFinishedAt)} UTC${took}`
        : `Idle · last plan at ${formatTime(status.lastFinishedAt)} UTC${took}`}
    </p>
  );
}
