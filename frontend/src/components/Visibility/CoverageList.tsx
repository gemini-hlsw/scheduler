import { useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@apollo/client";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ObservationStatus } from "@/gql/graphql";
import { observationCoverageQuery, REALTIME_CONTEXT } from "./query";
import { SITES, formatReason } from "./format";

const PAGE_SIZE = 50;

type CoverageRow = {
  observationId: string;
  programLabel: string;
  site?: string | null;
  targetName?: string | null;
  status: ObservationStatus;
  reason?: string | null;
};

const STATUSES: { value: ObservationStatus | "ALL"; label: string }[] = [
  { value: "ALL", label: "All" },
  { value: "MISSING", label: "Missing" },
  { value: "PENDING", label: "Being updated" },
  { value: "STORED", label: "Stored" },
  { value: "SKIPPED", label: "Not applicable" },
];

function StatusBadge({ status }: { status: ObservationStatus }) {
  if (status === "STORED")
    return (
      <Badge className="bg-emerald-600 text-white dark:bg-emerald-500">
        Stored
      </Badge>
    );
  if (status === "MISSING") return <Badge variant="destructive">Missing</Badge>;
  if (status === "PENDING")
    return (
      <Badge className="bg-amber-600 text-white dark:bg-amber-500">
        Updating
      </Badge>
    );
  return <Badge variant="secondary">N/A</Badge>;
}

function FilterButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <Button
      variant={active ? "default" : "outline"}
      size="xs"
      onClick={onClick}
    >
      {children}
    </Button>
  );
}

export default function CoverageList() {
  const [status, setStatus] = useState<ObservationStatus | "ALL">("MISSING");
  const [site, setSite] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);

  // Filters drive the server-side query; nothing is filtered client-side, so a
  // page costs the same whether there are 500 rows or 50,000.
  const { data, loading, error } = useQuery(observationCoverageQuery, {
    context: REALTIME_CONTEXT,
    fetchPolicy: "no-cache",
    variables: {
      nightDate: null,
      status: status === "ALL" ? null : status,
      site,
      programLabel: null,
      search: search.trim() || null,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    },
  });

  const result = data?.observationCoverage;
  const total = result?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Group the page by program so a long list reads as a handful of programs
  // rather than a wall of rows.
  const groups = useMemo(() => {
    const rows = result?.observations ?? [];
    const byProgram = new Map<string, CoverageRow[]>();
    for (const row of rows) {
      const existing = byProgram.get(row.programLabel);
      if (existing) existing.push(row);
      else byProgram.set(row.programLabel, [row]);
    }
    return [...byProgram.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [result]);

  function reset(update: () => void) {
    update();
    setPage(0);
  }

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-3 p-3",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <div className="flex flex-row gap-2 items-center flex-wrap">
        <h1 className="font-bold">Observations</h1>
        <div className="flex flex-row gap-1 flex-wrap">
          {STATUSES.map((option) => (
            <FilterButton
              key={option.value}
              active={status === option.value}
              onClick={() => reset(() => setStatus(option.value))}
            >
              {option.label}
            </FilterButton>
          ))}
        </div>
        <div className="flex flex-row gap-1">
          <FilterButton active={site === null} onClick={() => reset(() => setSite(null))}>
            Both
          </FilterButton>
          {SITES.map((key) => (
            <FilterButton
              key={key}
              active={site === key}
              onClick={() => reset(() => setSite(key))}
            >
              {key}
            </FilterButton>
          ))}
        </div>
        <Input
          className="h-6 w-52 text-xs"
          placeholder="Search reference label…"
          value={search}
          onChange={(e) => reset(() => setSearch(e.target.value))}
        />
      </div>

      {loading && (
        <span className="text-sm text-muted-foreground">Loading…</span>
      )}
      {error && <span className="text-sm text-destructive">{error.message}</span>}

      {result && !error && total === 0 && (
        <span className="text-sm text-muted-foreground">
          Nothing matches these filters.
        </span>
      )}

      {result && !error && total > 0 && (
        <>
          <Accordion
            type="multiple"
            defaultValue={groups.map(([label]) => label)}
            className="flex flex-col gap-1"
          >
            {groups.map(([programLabel, rows]) => (
              <AccordionItem key={programLabel} value={programLabel}>
                <AccordionTrigger className="py-1">
                  <span className="flex flex-row gap-2 items-center">
                    <span className="font-bold">{programLabel}</span>
                    <span className="text-xs text-muted-foreground">
                      {rows.length} on this page
                    </span>
                  </span>
                </AccordionTrigger>
                <AccordionContent>
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
                        <TableHead>Site</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Reason</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {rows.map((row) => (
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
                          <TableCell>{row.site ?? "—"}</TableCell>
                          <TableCell>
                            <StatusBadge status={row.status} />
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {formatReason(row.reason)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>

          <div className="flex flex-row gap-2 items-center">
            <span className="text-xs text-muted-foreground">
              {total} observation{total === 1 ? "" : "s"} · page {page + 1} of{" "}
              {pageCount}
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
