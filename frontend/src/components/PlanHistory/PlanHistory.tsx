import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@apollo/client";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { GeneratedPlansQuery } from "@/gql/graphql";
import type { NightPlanType } from "@/types";
import Result from "../Operation/Result";
import { REALTIME_CONTEXT } from "../Visibility/query";
import { formatUtcDateTime, formatUtcTime } from "../Visibility/format";
import { nightPlansQuery, planContentQuery, planNightsQuery } from "./query";
import { STATUS_STYLE, statusStyle } from "./status";
import { NightPicker } from "./NightPicker";

type PlanSummary = GeneratedPlansQuery["generatedPlans"][number];

const MODES = [
  { value: "", label: "All" },
  { value: "operation", label: "Operation" },
  { value: "simulation", label: "Simulation" },
];

/** How many recent nights are offered as one-click shortcuts. */
const RECENT_NIGHTS = 10;

function StatusBadge({ plan }: { plan: PlanSummary }) {
  const style = statusStyle(plan.status);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap",
        style.badge,
      )}
      title={style.description}
    >
      {style.label}
    </span>
  );
}

/**
 * The night's plans on a line, placed by when each was made, coloured by what
 * became of it. Mirrors the plan timeline of the Operation tab.
 */
function PlanTimeline({
  plans,
  selectedId,
  onSelect,
}: {
  plans: PlanSummary[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const times = plans.map((p) => new Date(p.createdAt).getTime());
  const first = Math.min(...times);
  const span = Math.max(...times) - first;

  return (
    <div className="flex flex-col gap-1">
      <div className="h-8 relative mx-3">
        <div className="absolute top-4 w-full border-t border-black/30 dark:border-white/30" />
        {plans.map((plan, i) => {
          // A single plan, or several made at once, sit in the middle.
          const pos = span > 0 ? ((times[i] - first) / span) * 100 : 50;
          const selected = plan.id === selectedId;
          return (
            <Tooltip key={plan.id}>
              <TooltipTrigger
                aria-label={`${formatUtcTime(plan.createdAt)} ${statusStyle(plan.status).label}: ${plan.eventDescription}`}
                onClick={() => onSelect(plan.id)}
                className={cn(
                  "absolute top-1 w-6 h-6 -ml-3 rounded-full cursor-pointer",
                  "border-2 transition-transform hover:scale-110",
                  statusStyle(plan.status).dot,
                  selected
                    ? "border-fuchsia-600 dark:border-fuchsia-400 scale-110"
                    : "border-white dark:border-black",
                )}
                style={{ left: `${pos}%` }}
              />
              <TooltipContent>
                <p className="font-semibold">
                  {formatUtcTime(plan.createdAt)} ·{" "}
                  {statusStyle(plan.status).label}
                </p>
                <p>{plan.eventDescription}</p>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
      <div className="flex flex-row justify-between text-xs text-muted-foreground">
        <span>{formatUtcTime(plans[0].createdAt)}</span>
        {plans.length > 1 && (
          <span>{formatUtcTime(plans.at(-1)!.createdAt)}</span>
        )}
      </div>
      <div className="flex flex-row flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {Object.entries(STATUS_STYLE).map(([key, style]) => (
          <span key={key} className="inline-flex items-center gap-1">
            <span
              className={cn("inline-block w-2.5 h-2.5 rounded-full", style.dot)}
            />
            {style.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function PlanList({
  plans,
  selectedId,
  onSelect,
}: {
  plans: PlanSummary[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow
          className={cn("dark:bg-white/20 bg-black/20", "*:h-6 *:font-bold")}
        >
          <TableHead>Made (UTC)</TableHead>
          <TableHead>Trigger</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Decided</TableHead>
          <TableHead>Mode</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {plans.map((plan) => (
          <TableRow
            key={plan.id}
            onClick={() => onSelect(plan.id)}
            aria-selected={plan.id === selectedId}
            className={cn(
              "cursor-pointer",
              plan.id === selectedId &&
                "bg-fuchsia-500/15 hover:bg-fuchsia-500/20",
            )}
          >
            <TableCell>{formatUtcTime(plan.createdAt)}</TableCell>
            <TableCell className="whitespace-normal">
              {plan.eventDescription}
              {plan.eventSite && (
                <span className="text-muted-foreground">
                  {" "}
                  · {plan.eventSite}
                </span>
              )}
            </TableCell>
            <TableCell>
              <StatusBadge plan={plan} />
            </TableCell>
            <TableCell>
              {plan.decidedBy
                ? `${plan.decidedBy}, ${formatUtcTime(plan.decidedAt)}`
                : "—"}
            </TableCell>
            <TableCell className="capitalize">{plan.mode}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function SelectedPlan({ plan }: { plan: PlanSummary }) {
  const { data, loading, error } = useQuery(planContentQuery, {
    context: REALTIME_CONTEXT,
    variables: { planId: plan.id },
  });
  const nightPlans: NightPlanType[] =
    data?.generatedPlanContent?.nightPlans?.nightTimeline ?? [];

  return (
    <section aria-label="Selected plan" className="flex flex-col gap-2">
      <div className="flex flex-row flex-wrap items-center gap-x-3 gap-y-1 text-sm">
        <StatusBadge plan={plan} />
        <span>
          Made {formatUtcDateTime(plan.createdAt)} for{" "}
          <span className="font-semibold">{plan.eventDescription}</span>
        </span>
        <span className="text-muted-foreground">
          {plan.decidedBy
            ? `${statusStyle(plan.status).label} by ${plan.decidedBy} at ${formatUtcDateTime(plan.decidedAt)}`
            : statusStyle(plan.status).description}
        </span>
        <span className="text-muted-foreground ml-auto text-xs">
          {plan.mode} · {plan.appVersion}
        </span>
      </div>
      {loading && (
        <span className="text-sm text-muted-foreground">Loading plan…</span>
      )}
      {error && (
        <span className="text-sm text-destructive">{error.message}</span>
      )}
      {!loading && !error && nightPlans.length === 0 && (
        <span className="text-sm text-muted-foreground">
          This plan is empty.
        </span>
      )}
      {nightPlans.length > 0 && <Result plans={nightPlans} label="Plan" />}
    </section>
  );
}

/**
 * Every plan the scheduler generated on a night, in order, with what became of
 * each: in effect, accepted, discarded, superseded or expired. Reads the
 * generated_plans table through the realtime backend, which holds the Sight DB.
 */
export default function PlanHistory() {
  const [mode, setMode] = useState("");
  const [nightDate, setNightDate] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const modeVariable = mode || null;

  const nights = useQuery(planNightsQuery, {
    context: REALTIME_CONTEXT,
    fetchPolicy: "network-only",
    variables: { mode: modeVariable },
  });
  const recentNights = nights.data?.generatedPlanNights ?? [];

  // Open on the latest night with plans, until the user picks another.
  useEffect(() => {
    if (nightDate === null && recentNights.length > 0) {
      setNightDate(recentNights[0].nightDate);
    }
  }, [nightDate, recentNights]);

  const night = useQuery(nightPlansQuery, {
    context: REALTIME_CONTEXT,
    fetchPolicy: "network-only",
    skip: nightDate === null,
    variables: { nightDate: nightDate ?? "", mode: modeVariable },
  });
  const plans = useMemo(() => night.data?.generatedPlans ?? [], [night.data]);

  // Show the night's latest plan unless the selected one is still listed.
  useEffect(() => {
    if (plans.length === 0) {
      setSelectedId(null);
    } else if (!plans.some((p) => p.id === selectedId)) {
      setSelectedId(plans.at(-1)!.id);
    }
  }, [plans, selectedId]);

  const selected = plans.find((p) => p.id === selectedId) ?? null;

  return (
    <div className="flex flex-col gap-3">
      <div
        className={cn(
          "border rounded-md p-3 flex flex-col gap-3",
          "dark:bg-white/20 bg-black/10",
        )}
      >
        <div className="flex flex-row flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1 text-sm">
            <Label htmlFor="plan-history-night">
              Night (UTC date of its morning)
            </Label>
            <NightPicker
              nightDate={nightDate}
              onChange={setNightDate}
              nightsWithPlans={recentNights.map((n) => n.nightDate)}
            />
          </div>
          <div className="flex flex-col gap-1 text-sm">
            Server mode
            <div className="flex flex-row gap-1">
              {MODES.map((m) => (
                <Button
                  key={m.value}
                  size="sm"
                  variant={mode === m.value ? "default" : "outline"}
                  onClick={() => setMode(m.value)}
                >
                  {m.label}
                </Button>
              ))}
            </div>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="ml-auto"
            onClick={() => {
              nights.refetch();
              if (nightDate !== null) night.refetch();
            }}
          >
            Refresh
          </Button>
        </div>
        {recentNights.length > 0 && (
          <div className="flex flex-row flex-wrap items-center gap-1 text-xs">
            <span className="text-muted-foreground mr-1">Recent nights:</span>
            {recentNights.slice(0, RECENT_NIGHTS).map((n) => (
              <Button
                key={n.nightDate}
                size="sm"
                variant={n.nightDate === nightDate ? "default" : "ghost"}
                className="h-6 px-2 text-xs"
                onClick={() => setNightDate(n.nightDate)}
              >
                {n.nightDate} ({n.planCount})
              </Button>
            ))}
          </div>
        )}
        {nights.error && (
          <span className="text-sm text-destructive">
            {nights.error.message}
          </span>
        )}
      </div>

      {night.loading && (
        <span className="text-sm text-muted-foreground">Loading plans…</span>
      )}
      {night.error && (
        <span className="text-sm text-destructive">{night.error.message}</span>
      )}
      {!nights.loading && nightDate === null && !nights.error && (
        <span className="text-sm text-muted-foreground">
          No plans have been stored yet.
        </span>
      )}
      {nightDate !== null &&
        !night.loading &&
        !night.error &&
        plans.length === 0 && (
          <span className="text-sm text-muted-foreground">
            No plans stored for the night of {nightDate}.
          </span>
        )}

      {plans.length > 0 && (
        <>
          <div
            className={cn(
              "border rounded-md p-3 flex flex-col gap-3",
              "dark:bg-white/20 bg-black/10",
            )}
          >
            <h2 className="font-bold">
              {plans.length} plan{plans.length === 1 ? "" : "s"} for the night
              of {nightDate}
            </h2>
            <PlanTimeline
              plans={plans}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
            <PlanList
              plans={plans}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          </div>
          {selected && <SelectedPlan key={selected.id} plan={selected} />}
        </>
      )}
    </div>
  );
}
