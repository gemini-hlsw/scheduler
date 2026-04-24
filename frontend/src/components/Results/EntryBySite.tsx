import { useEffect, useState } from "react";
import { TimeEntriesBySite, TimeEntryType } from "../../types";
import TimeEntry from "./TimeEntry";
import { FaCloud, FaCog } from "react-icons/fa";
import { LuTimerOff } from "react-icons/lu";
import { Badge } from "@/components/ui/badge";
import { TimelineBullets } from "./TimelineBullets";

export default function EntryBySite({
  entryBySite,
}: {
  entryBySite: TimeEntriesBySite;
}) {
  const [selectedEntry, setSelectedEntry] = useState<TimeEntryType>(
    entryBySite.timeEntries[0] ?? ({} as TimeEntryType)
  );

  useEffect(() => {
    setSelectedEntry(entryBySite.timeEntries[0] ?? ({} as TimeEntryType));
  }, [entryBySite]);

  const timelineDate =
    entryBySite.mornTwilight.substring(
      0,
      entryBySite.mornTwilight.indexOf("T")
    ) ?? "";

  return (
    <div className="flex flex-col gap-2">
      <h4 className="font-bold">Timeline {timelineDate}</h4>
      <div className="flex flex-row gap-2">
        <Badge className={"text-white text-sm bg-red-500"}>
          <FaCog />
          Faults time: {entryBySite.timeLosses.fault.toFixed(2)}
        </Badge>
        <Badge className={"text-white text-sm bg-blue-500"}>
          <FaCloud />
          Weather time: {entryBySite.timeLosses.weather.toFixed(2)}
        </Badge>
        <Badge className={"text-white text-sm bg-purple-500"}>
          <LuTimerOff />
          Unscheduled time: {entryBySite.timeLosses.unschedule.toFixed(2)}
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
