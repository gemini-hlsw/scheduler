import { PlanPerSite, Visit } from "../../types";
import AltAzPlot from "../SchedulerPlot/SchedulerPlot";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { ObsClassBadge } from "../Results/ObsClassBadge";

export default function RtPlan({ plan }: { plan: PlanPerSite }) {
  function parseToVisitForPlot(visits: Visit[]) {
    return visits.map((visit: Visit) => ({
      startDate: new Date(visit.startTime),
      endDate: new Date(visit.endTime),
      yPoints: visit.altitude,
      label: visit.obsId,
      instrument: visit.instrument,
    }));
  }
  const tz = plan.site === "GN" ? "Pacific/Honolulu" : "America/Santiago";
  const formatScore = (score: number) => {
    return score.toFixed(2);
  };

  function fractionToPercentage(fraction: string): number {
    const parts = fraction.split("/");
    if (parts.length !== 2) {
      throw new Error("Invalid fraction format");
    }
    const numerator = parseFloat(parts[0]);
    const denominator = parseFloat(parts[1]);
    if (denominator === 0) {
      throw new Error("Denominator cannot be zero");
    }
    const percentage = (numerator / denominator) * 100;

    return percentage;
  }

  const scoreBodyTemplate = (visit: Visit) => {
    return formatScore(visit.score);
  };
  const peakScoreBodyTemplate = (visit: Visit) => {
    return formatScore(visit.peakScore);
  };

  const obsCompletionBodyTemplate = (visit: Visit) => {
    return `${visit.completion} (${fractionToPercentage(
      visit.completion
    ).toFixed(0)}%)`;
  };

  return (
    <>
      <AltAzPlot
        data={parseToVisitForPlot(plan.visits)}
        eveTwilight={plan.startTime}
        mornTwilight={plan.endTime}
        site={plan.site}
      />
      <Table>
        <TableHeader>
          <TableRow className={cn("*:h-6 *:font-bold")}>
            <TableHead>Observation Id</TableHead>
            <TableHead>Observation Class</TableHead>
            <TableHead>Start Time</TableHead>
            <TableHead>Atom Start</TableHead>
            <TableHead>Atom End</TableHead>
            <TableHead>Instrument</TableHead>
            <TableHead>FPU</TableHead>
            <TableHead>Grating</TableHead>
            <TableHead>Filters</TableHead>
            <TableHead>Cloud Cover</TableHead>
            <TableHead>Image Quality</TableHead>
            <TableHead>Obs Completion</TableHead>
            <TableHead>Peak Score</TableHead>
            <TableHead>Score</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {plan.visits.map((visit: Visit) => (
            <TableRow
              key={visit.obsId}
              className={cn(
                "odd:bg-muted/50 *:p-0 *:px-2",
                "dark:hover:bg-white/30 hover:bg-black/30"
              )}
            >
              <TableCell>{visit.obsId}</TableCell>
              <TableCell>
                <ObsClassBadge obsClass={visit.obsClass} />
              </TableCell>
              <TableCell>
                {new Date(visit.startTime).toLocaleString("en-UK", {
                  timeZone: tz,
                })}
              </TableCell>
              <TableCell>{visit.atomStartIdx}</TableCell>
              <TableCell>{visit.atomEndIdx}</TableCell>
              <TableCell>{visit.instrument}</TableCell>
              <TableCell>{visit.fpu}</TableCell>
              <TableCell>{visit.disperser}</TableCell>
              <TableCell>{visit.filters}</TableCell>
              <TableCell>{visit.requiredConditions.cc}</TableCell>
              <TableCell>{visit.requiredConditions.iq}</TableCell>
              <TableCell>{obsCompletionBodyTemplate(visit)}</TableCell>
              <TableCell>{peakScoreBodyTemplate(visit)}</TableCell>
              <TableCell>{scoreBodyTemplate(visit)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </>
  );
}
