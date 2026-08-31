import { TimeEntryType, Visit } from "../../types";
import NightPlanSummary from "./NightPlanSummary";
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
import { ObsClassBadge } from "./ObsClassBadge";
import { FaCheck } from "react-icons/fa";
import { AiOutlineLoading3Quarters } from "react-icons/ai";
import { IoHourglassOutline } from "react-icons/io5";

function VisitRow({
  visit,
  site,
  state,
  beingExecuted = false,
  drawTopBorder = false,
  drawBotBorder = false,
}: {
  visit: Visit;
  site: string;
  state: string;
  beingExecuted?: boolean;
  drawTopBorder?: boolean;
  drawBotBorder?: boolean;
}) {
  const tz = site === "GN" ? "Pacific/Honolulu" : "America/Santiago";

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

  const formatScore = (score: number) => {
    return score.toFixed(2);
  };

  const scoreBodyTemplate = (visit: Visit) => {
    return formatScore(visit.score);
  };
  const peakScoreBodyTemplate = (visit: Visit) => {
    return formatScore(visit.peakScore);
  };

  const obsCompletionBodyTemplate = (visit: Visit) => {
    return `${visit.completion} (${fractionToPercentage(
      visit.completion,
    ).toFixed(0)}%)`;
  };

  let icon = null;
  switch (state) {
    case "executed":
      icon = <FaCheck className="text-green-500" />;
      break;
    case "executing":
      icon = (
        <AiOutlineLoading3Quarters className="animate-spin text-yellow-500" />
      );
      break;
    case "scheduled":
      icon = <IoHourglassOutline className="text-yellow-500" />;
      break;
    default:
      icon = null;
  }

  return (
    <TableRow
      className={cn(
        beingExecuted
          ? "dark:bg-blue-700/50 bg-blue-400/50"
          : "odd:bg-black/10 dark:odd:bg-white/10",
        beingExecuted
          ? "hover:dark:bg-blue-400/50 hover:bg-blue-700/50"
          : "dark:hover:bg-white/30 hover:bg-black/30",
        "*:p-0 *:px-2",
        drawTopBorder ? "*:border-t-2 *:border-t-blue-500" : "",
        drawBotBorder ? "*:border-b-2 *:border-b-blue-500" : "",
      )}
    >
      <TableCell>{icon}</TableCell>
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
  );
}

export default function TimeEntry({
  timeEntry,
  mornTwilight,
  eveTwilight,
  site,
}: {
  timeEntry: TimeEntryType;
  mornTwilight: string;
  eveTwilight: string;
  site: string;
}) {
  function parseToVisitForPlot(visits: Visit[]) {
    return visits.map((visit: Visit) => ({
      startDate: new Date(visit.startTime),
      endDate: new Date(visit.endTime),
      yPoints: visit.altitude,
      label: visit.obsId,
      instrument: visit.instrument,
      atomTimes: visit.atomTimes,
    }));
  }

  if (!timeEntry || !timeEntry.plan) return <div>No plan found</div>;

  const eventTime = new Date(timeEntry.event.time);
  const executedVisits = [];
  const beingExecutedVisits = [];
  const scheduledVisits = [];
  const visitRows = [];

  for (const visit of timeEntry.plan.visits) {
    if (new Date(visit.endTime) < eventTime) {
      executedVisits.push(visit);
    } else if (new Date(visit.startTime) > eventTime) {
      scheduledVisits.push(visit);
    } else {
      beingExecutedVisits.push(visit);
    }
  }

  for (let i = 0; i < executedVisits.length; i++) {
    if (i === executedVisits.length - 1) {
      visitRows.push(
        <VisitRow
          key={executedVisits[i].startTime}
          visit={executedVisits[i]}
          site={site}
          state="executed"
          drawBotBorder={beingExecutedVisits.length === 0}
        />,
      );
    } else {
      visitRows.push(
        <VisitRow
          key={executedVisits[i].startTime}
          visit={executedVisits[i]}
          site={site}
          state="executed"
        />,
      );
    }
  }

  for (let i = 0; i < beingExecutedVisits.length; i++) {
    visitRows.push(
      <VisitRow
        key={beingExecutedVisits[i].startTime}
        visit={beingExecutedVisits[i]}
        site={site}
        state="executing"
        beingExecuted={true}
      />,
    );
  }

  for (let i = 0; i < scheduledVisits.length; i++) {
    if (i === 0) {
      visitRows.push(
        <VisitRow
          key={scheduledVisits[i].startTime}
          visit={scheduledVisits[i]}
          site={site}
          state="scheduled"
          drawTopBorder={
            beingExecutedVisits.length === 0 && executedVisits.length === 0
          }
        />,
      );
    } else {
      visitRows.push(
        <VisitRow
          key={scheduledVisits[i].startTime}
          visit={scheduledVisits[i]}
          site={site}
          state="scheduled"
        />,
      );
    }
  }

  return (
    <div>
      <NightPlanSummary
        timestats={timeEntry.timestats}
        nightState={timeEntry.plan.nightStats}
        nightTitle={timeEntry.event.description}
        nightConditions={timeEntry.plan.nightConditions}
      />
      <AltAzPlot
        data={parseToVisitForPlot(timeEntry.plan.visits)}
        event={timeEntry.event}
        eveTwilight={eveTwilight}
        mornTwilight={mornTwilight}
        site={site}
        closureWindows={timeEntry.timelossWindows}
      />
      <Table>
        <TableHeader>
          <TableRow
            className={cn("dark:bg-white/20 bg-black/20", "*:h-6 *:font-bold")}
          >
            <TableHead></TableHead>
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
        <TableBody>{visitRows}</TableBody>
      </Table>
      <Table>
        <TableHeader>
          <TableRow
            className={cn("dark:bg-white/20 bg-black/20", "*:h-6 *:font-bold")}
          >
            <TableHead>Program Id</TableHead>
            <TableHead>Completition</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Object.keys(timeEntry.plan.nightStats.programCompletion).length >
          0 ? (
            Object.keys(timeEntry.plan.nightStats.programCompletion).map(
              (progId: string) => (
                <TableRow
                  key={`${timeEntry.plan.startTime}-${progId}`}
                  className={cn(
                    "odd:bg-muted/50 *:p-0 *:px-2",
                    "dark:hover:bg-white/30 hover:bg-black/30",
                  )}
                >
                  <TableCell>{progId}</TableCell>
                  <TableCell>
                    {timeEntry.plan.nightStats.programCompletion[progId]}
                  </TableCell>
                </TableRow>
              ),
            )
          ) : (
            <TableRow
              className={cn(
                "odd:bg-muted/50 *:p-0 *:px-2",
                "dark:hover:bg-white/30 hover:bg-black/30",
              )}
            >
              <TableCell>No available options</TableCell>
              <TableCell></TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
