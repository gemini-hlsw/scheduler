import { useLayoutEffect, useRef, useState } from "react";
import { makeVar, useReactiveVar, useSubscription } from "@apollo/client";
import { cn } from "@/lib/utils";
import { isRealtimeConnectedVar } from "@/apollo-client";
import { ConnectionBadge } from "../ui/connectionBadge";
import { Badge } from "../ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import { schedulerEventsSubscription } from "./subscription";
import type { SchedulerEventsSubscription } from "@/gql/graphql";

type SchedulerEvent = SchedulerEventsSubscription["schedulerEvents"];

// Same bound as the server history, so the list does not grow all night.
const MAX_EVENTS = 200;
// Rows shown at once; older events are reached by scrolling.
const VISIBLE_EVENTS = 3;

// WeatherChangeEvent -> Weather Change
const formatType = (eventType: string) =>
  eventType
    .replace(/Event$/, "")
    .replace(/([a-z])([A-Z])/g, "$1 $2");

// Outside the component so the events survive switching tabs; only a page
// reload clears them. Newest first.
const schedulerEventsVar = makeVar<SchedulerEvent[]>([]);

const formatTime = (time?: string | null) =>
  time ? new Date(time).toISOString().slice(11, 19) : "";

export default function SchedulerEvents() {
  const isOnline = useReactiveVar(isRealtimeConnectedVar);
  const events = useReactiveVar(schedulerEventsVar);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [maxHeight, setMaxHeight] = useState<number>();

  // Descriptions wrap, so rows differ in height: measure the header and the
  // first rows rather than assume a fixed row height.
  useLayoutEffect(() => {
    const table = scrollRef.current?.querySelector("table");
    if (!table) return;
    const rows = [...table.rows].slice(0, VISIBLE_EVENTS + 1);
    setMaxHeight(rows.reduce((height, row) => height + row.offsetHeight, 0));
  }, [events]);

  // onData, not data: the server replays its history in a burst on connect,
  // and React would only render the last one.
  const { error } = useSubscription(schedulerEventsSubscription, {
    context: { clientName: "realtimeClient" },
    onData: ({ data }) => {
      const event = data.data?.schedulerEvents;
      if (!event) return;
      const current = schedulerEventsVar();
      // The history is replayed again on every (re)subscribe, e.g. when coming
      // back to this tab.
      if (current.some((e) => e.id === event.id)) return;
      schedulerEventsVar([event, ...current].slice(0, MAX_EVENTS));
    },
  });

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3",
        "dark:bg-white/20 bg-black/10",
      )}
    >
      <h1 className="font-bold flex justify-between gap-2">
        <span>Scheduler Events</span>
        <ConnectionBadge isOnline={isOnline} />
      </h1>
      {error && <p>Error loading scheduler events: {error.message}</p>}
      {!error && events.length === 0 && (
        <p className="text-sm opacity-70">No events received yet.</p>
      )}
      {events.length > 0 && (
        <div ref={scrollRef}>
          <Table
            containerProps={{ className: "overflow-y-auto", style: { maxHeight } }}
          >
            {/* Pinned while the rows scroll. It needs an opaque background:
                the page colour, tinted like the panel so it looks the same.
                The collapsed table border would scroll away, so the bottom
                line is a shadow. */}
            <TableHeader className="sticky top-0 z-10 bg-background shadow-[0_1px_0_var(--border)]">
              <TableRow className="*:px-2 border-0 dark:bg-white/20 bg-black/10 hover:bg-black/10 dark:hover:bg-white/20">
                <TableHead>Time (UTC)</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Sites</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Plan</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {events.map((event) => (
                <TableRow
                  key={event.id}
                  className={cn(
                    "odd:bg-muted/50 *:px-2",
                    "dark:hover:bg-white/30 hover:bg-black/30",
                  )}
                >
                  {/* Events without a time of their own (e.g. a start-of-night
                      plan request) show when they were received. */}
                  <TableCell>
                    {formatTime(event.eventTime ?? event.receivedAt)}
                  </TableCell>
                  <TableCell>{formatType(event.eventType)}</TableCell>
                  <TableCell>{event.sites.join(", ") || "—"}</TableCell>
                  <TableCell className="whitespace-normal">
                    {event.description}
                  </TableCell>
                  <TableCell>
                    {event.triggeredPlan ? (
                      <Badge className="bg-green-600 text-white">New plan</Badge>
                    ) : (
                      <Badge variant="outline">Recorded</Badge>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
