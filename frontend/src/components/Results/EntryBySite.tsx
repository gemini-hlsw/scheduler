import { useEffect, useState } from "react";
import { TimeEntriesBySite, TimeEntryType } from "../../types";
import TimeEntry from "./TimeEntry";
import { FaClock, FaCloud, FaCog } from "react-icons/fa";
import { TbTelescopeOff } from "react-icons/tb";
import { LuTimerOff } from "react-icons/lu";
import { Badge } from "@/components/ui/badge";
import { TimelineBullets } from "./TimelineBullets";

export default function EntryBySite({
  entryBySite,
}: {
  entryBySite: TimeEntriesBySite;
}) {
  const [selectedEntry, setSelectedEntry] = useState<TimeEntryType>(
    entryBySite.timeEntries[0] ?? ({} as TimeEntryType),
  );

  useEffect(() => {
    setSelectedEntry(entryBySite.timeEntries[0] ?? ({} as TimeEntryType));
  }, [entryBySite]);

  const timelineDate =
    entryBySite.mornTwilight.substring(
      0,
      entryBySite.mornTwilight.indexOf("T"),
    ) ?? "";

  return (
    <div className="flex flex-col gap-2">
      <h4 className="font-bold">Timeline {timelineDate}</h4>
      <div className="flex flex-row gap-2">
        <Badge className={"text-white text-sm bg-gray-500"}>
          <FaClock />
          Total night time: {entryBySite.timestats.nightLength.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-green-500"}>
          <FaClock />
          Observed time: {entryBySite.timestats.observed.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-yellow-500"}>
          <FaClock />
          Scheduled time: {entryBySite.timestats.scheduled.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-orange-500"}>
          <FaCog />
          Faults time: {entryBySite.timestats.fault.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-blue-500"}>
          <FaCloud />
          Weather time: {entryBySite.timestats.weather.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-red-500"}>
          <TbTelescopeOff />
          Telescope closed time: {entryBySite.timestats.closed.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-purple-500"}>
          <LuTimerOff />
          Unscheduled time: {entryBySite.timestats.unscheduled.toFixed(0)}
        </Badge>
      </div>
      <TimelineBullets
        date={timelineDate}
        timeline={entryBySite?.timeEntries}
        selectedEntry={selectedEntry}
        setSelectedEntry={setSelectedEntry}
      />
      <TimeEntry
        timeEntry={selectedEntry}
        eveTwilight={entryBySite.eveTwilight}
        mornTwilight={entryBySite.mornTwilight}
        site={entryBySite.site}
      />
    </div>
  );
}
