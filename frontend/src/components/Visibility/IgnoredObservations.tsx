import { useMemo, useState } from "react";
import { useMutation, useQuery, useSubscription } from "@apollo/client";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { observationCoverageQuery, REALTIME_CONTEXT } from "./query";
import { setObservationsIgnoredMutation } from "./mutation";
import { ignoredObservationsSubscription } from "./subscription";
import { SITES, formatAgo, formatUtcDateTime } from "./format";

const PAGE_SIZE = 20;

const CARD = cn(
  "border rounded-md flex flex-col gap-3 p-3",
  "dark:bg-white/20 bg-black/10"
);
const HEADER_ROW = cn("dark:bg-white/20 bg-black/20", "*:h-6 *:font-bold");
const BODY_ROW = cn(
  "odd:bg-muted/50 *:p-0 *:px-2",
  "dark:hover:bg-white/30 hover:bg-black/30"
);

type IgnoredRow = { observationId: string; ignoredAt: string };

/**
 * Observations left out of the plans: the list with an undo, and a finder over
 * the same observations the Coverage tab lists to add more.
 */
export default function IgnoredObservations() {
  const [site, setSite] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  // The list is shared by every browser: a change goes up as a mutation and
  // comes back to all of them through the subscription, which also sends the
  // current list when it connects. Both carry the whole list, so the latest wins.
  const [ignored, setIgnoredRows] = useState<IgnoredRow[]>([]);
  const live = useSubscription(ignoredObservationsSubscription, {
    context: { ...REALTIME_CONTEXT, queryDeduplication: false },
    onData: ({ data }) => {
      const rows = data.data?.ignoredObservationsUpdates;
      if (rows) setIgnoredRows(rows);
    },
  });
  const [setIgnored, mutation] = useMutation(setObservationsIgnoredMutation, {
    context: REALTIME_CONTEXT,
    fetchPolicy: "no-cache",
    onCompleted: (data) => setIgnoredRows(data.setObservationsIgnored),
  });

  // Served from the backend's last coverage read, so searching does not re-read the ODB.
  const finder = useQuery(observationCoverageQuery, {
    context: REALTIME_CONTEXT,
    fetchPolicy: "no-cache",
    variables: {
      nightDate: null,
      status: null,
      site,
      programLabel: null,
      search: search.trim() || null,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    },
  });

  const ignoredIds = useMemo(
    () => new Set(ignored.map((row) => row.observationId)),
    [ignored]
  );

  const result = finder.data?.observationCoverage;
  const total = result?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const error = live.error ?? mutation.error;

  function change(observationId: string, isIgnored: boolean) {
    // A rejection is already shown through mutation.error.
    setIgnored({
      variables: { observationIds: [observationId], ignored: isIgnored },
    }).catch(() => {});
  }

  function reset(update: () => void) {
    update();
    setPage(0);
  }

  return (
    <div className="flex flex-col gap-3">
      <div className={CARD}>
        <div className="flex flex-row gap-2 items-center flex-wrap">
          <h1 className="font-bold">Ignored observations</h1>
          <span className="text-xs text-muted-foreground">
            {ignored.length} left out of the plans · applies from the next plan
          </span>
        </div>

        {live.loading && (
          <span className="text-sm text-muted-foreground">Loading…</span>
        )}
        {error && (
          <span className="text-sm text-destructive">{error.message}</span>
        )}
        {!live.loading && !live.error && ignored.length === 0 && (
          <span className="text-sm text-muted-foreground">
            Nothing is ignored. Find an observation below to leave it out of
            the plans.
          </span>
        )}

        {ignored.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow className={HEADER_ROW}>
                <TableHead>Observation</TableHead>
                <TableHead>Ignored since</TableHead>
                <TableHead className="text-right">Undo</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ignored.map((row) => (
                <TableRow key={row.observationId} className={BODY_ROW}>
                  <TableCell className="font-mono text-xs">
                    {row.observationId}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {formatUtcDateTime(row.ignoredAt)} ·{" "}
                    {formatAgo(row.ignoredAt)}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="outline"
                      size="xs"
                      disabled={mutation.loading}
                      onClick={() => change(row.observationId, false)}
                    >
                      Stop ignoring
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      <div className={CARD}>
        <div className="flex flex-row gap-2 items-center flex-wrap">
          <h1 className="font-bold">Find an observation</h1>
          <div className="flex flex-row gap-1">
            <Button
              variant={site === null ? "default" : "outline"}
              size="xs"
              onClick={() => reset(() => setSite(null))}
            >
              Both
            </Button>
            {SITES.map((key) => (
              <Button
                key={key}
                variant={site === key ? "default" : "outline"}
                size="xs"
                onClick={() => reset(() => setSite(key))}
              >
                {key}
              </Button>
            ))}
          </div>
          <Input
            className="h-6 w-52 text-xs"
            placeholder="Search reference label…"
            value={search}
            onChange={(e) => reset(() => setSearch(e.target.value))}
          />
        </div>

        {finder.loading && (
          <span className="text-sm text-muted-foreground">Loading…</span>
        )}
        {finder.error && (
          <span className="text-sm text-destructive">
            {finder.error.message}
          </span>
        )}
        {result && !finder.error && total === 0 && (
          <span className="text-sm text-muted-foreground">
            Nothing matches this search.
          </span>
        )}

        {result && !finder.error && total > 0 && (
          <>
            <Table>
              <TableHeader>
                <TableRow className={HEADER_ROW}>
                  <TableHead>Observation</TableHead>
                  <TableHead>Target</TableHead>
                  <TableHead>Site</TableHead>
                  <TableHead className="text-right">Ignore</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.observations.map((row) => {
                  const isIgnored = ignoredIds.has(row.observationId);
                  return (
                    <TableRow key={row.observationId} className={BODY_ROW}>
                      <TableCell className="font-mono text-xs">
                        {row.observationId}
                      </TableCell>
                      <TableCell>{row.targetName ?? "—"}</TableCell>
                      <TableCell>{row.site ?? "—"}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant={isIgnored ? "outline" : "default"}
                          size="xs"
                          disabled={mutation.loading}
                          onClick={() => change(row.observationId, !isIgnored)}
                        >
                          {isIgnored ? "Stop ignoring" : "Ignore"}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>

            <div className="flex flex-row gap-2 items-center">
              <span className="text-xs text-muted-foreground">
                {total} observation{total === 1 ? "" : "s"} · page {page + 1}{" "}
                of {pageCount}
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
    </div>
  );
}
