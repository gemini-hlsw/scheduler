import { useState } from "react";
import { useQuery } from "@apollo/client";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { visibleObservationsQuery, REALTIME_CONTEXT } from "./query";
import { formatMinutes, formatUtcTime, SITES, type SiteKey } from "./format";

const PAGE_SIZE = 50;

function SitePanel({ site }: { site: SiteKey }) {
  const [page, setPage] = useState(0);

  const { data, loading, error } = useQuery(visibleObservationsQuery, {
    context: REALTIME_CONTEXT,
    fetchPolicy: "no-cache",
    variables: {
      site,
      nightDate: null,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
      // Observations with no visibility at all are not useful here: the
      // question is what *can* be observed.
      minRemainingMinutes: 1,
    },
  });

  const result = data?.visibleObservations;
  const total = result?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-row gap-3 items-center flex-wrap">
        {result?.nightDate && (
          <span className="text-xs text-muted-foreground">
            night {result.nightDate} (UTC)
          </span>
        )}
        <span className="text-xs text-muted-foreground">
          {total} observation{total === 1 ? "" : "s"} visible
        </span>
      </div>

      {loading && <span className="text-sm text-muted-foreground">Loading…</span>}
      {error && <span className="text-sm text-destructive">{error.message}</span>}

      {result && !error && total === 0 && (
        <span className="text-sm text-muted-foreground">
          Nothing stored for this night yet.
        </span>
      )}

      {result && !error && total > 0 && (
        <>
          <Table>
            <TableHeader>
              <TableRow
                className={cn(
                  "dark:bg-white/20 bg-black/20",
                  "*:h-6 *:font-bold"
                )}
              >
                <TableHead>Observation</TableHead>
                <TableHead>Target</TableHead>
                <TableHead>Visible tonight</TableHead>
                <TableHead>Left from now</TableHead>
                <TableHead>Windows (UTC)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {result.observations.map((row) => (
                <TableRow
                  key={row.observationId}
                  className={cn(
                    "odd:bg-muted/50 *:p-0 *:px-2",
                    "dark:hover:bg-white/30 hover:bg-black/30"
                  )}
                >
                  <TableCell className="font-mono text-xs">
                    {row.observationId}
                  </TableCell>
                  <TableCell>{row.targetName ?? "—"}</TableCell>
                  <TableCell>{formatMinutes(row.remainingMinutes)}</TableCell>
                  <TableCell
                    className={cn(
                      row.remainingMinutesFromNow === 0 &&
                        "text-muted-foreground"
                    )}
                  >
                    {formatMinutes(row.remainingMinutesFromNow)}
                  </TableCell>
                  <TableCell className="text-xs">
                    {row.intervals
                      .map(
                        (interval) =>
                          `${formatUtcTime(interval.start)}–${formatUtcTime(
                            interval.end
                          )}`
                      )
                      .join(", ")}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          <div className="flex flex-row gap-2 items-center">
            <span className="text-xs text-muted-foreground">
              page {page + 1} of {pageCount} ·{" "}
              {formatMinutes(result.totalRemainingMinutes)} left on this page
            </span>
            <div className="ml-auto flex flex-row gap-1">
              <Button
                variant="outline"
                size="xs"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="xs"
                disabled={page + 1 >= pageCount}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function VisibleTonight() {
  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-3 p-3",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <h1 className="font-bold">Visible tonight</h1>
      {/* Each site resolves its own night: GN and GS roll over at different
          instants, so one shared date would be wrong for one of them. */}
      <Tabs defaultValue="GN">
        <TabsList>
          {SITES.map((site) => (
            <TabsTrigger key={site} value={site}>
              {site}
            </TabsTrigger>
          ))}
        </TabsList>
        {SITES.map((site) => (
          <TabsContent key={site} value={site}>
            <SitePanel site={site} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
