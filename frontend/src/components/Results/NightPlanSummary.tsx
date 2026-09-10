import { NightConditions, NightStats, TimeStats } from "../../types";
import { Badge } from "../ui/badge";
import { FaClock, FaCloud, FaCog } from "react-icons/fa";
import { TbTelescopeOff } from "react-icons/tb";
import { LuTimerOff } from "react-icons/lu";

export default function NightPlanSummary({
  nightState,
  timestats,
  nightConditions,
  nightTitle,
}: {
  nightState: NightStats;
  timestats: TimeStats;
  nightConditions: NightConditions;
  nightTitle: string;
}) {
  const completion = nightState.completionFraction;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="font-bold">{nightTitle}</h4>
      <div className="flex flex-row flex-wrap gap-2">
        <Badge className={"text-white text-sm bg-gray-500"}>
          <FaClock />
          Total night time: {timestats.nightLength.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-green-500"}>
          <FaClock />
          Observed time: {timestats.observed.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-yellow-500"}>
          <FaClock />
          Scheduled time: {timestats.scheduled.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-orange-500"}>
          <FaCog />
          Faults time: {timestats.fault.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-blue-500"}>
          <FaCloud />
          Weather time: {timestats.weather.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-red-500"}>
          <TbTelescopeOff />
          Telescope closed time: {timestats.closed.toFixed(0)}
        </Badge>
        <Badge className={"text-white text-sm bg-purple-500"}>
          <LuTimerOff />
          Unscheduled time: {timestats.unscheduled.toFixed(0)}
        </Badge>
        <Badge className="text-sm">Cloud Cover: {nightConditions.cc}</Badge>
        <Badge className="text-sm">Image Quality: {nightConditions.iq}</Badge>
        <Badge className="text-sm">ToOs: {nightState.nToos}</Badge>
        <Badge className="text-sm">
          Score: {nightState.planScore.toFixed(2)}
        </Badge>
        {completion[1] > 0 && (
          <Badge className="text-sm">Band 1: {completion[1]}</Badge>
        )}
        {completion[2] > 0 && (
          <Badge className="text-sm">Band 2: {completion[2]}</Badge>
        )}
        {completion[3] > 0 && (
          <Badge className="text-sm">Band 3: {completion[3]}</Badge>
        )}
        {completion[4] > 0 && (
          <Badge className="text-sm">Band 4: {completion[4]}</Badge>
        )}
      </div>
    </div>
  );
}
