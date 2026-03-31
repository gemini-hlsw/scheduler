import { NightConditions, NightStats } from "../../types";
import { Badge } from "../ui/badge";

export default function NightPlanSummary({
  nightState,
  nightConditions,
  nightTitle,
}: {
  nightState: NightStats;
  nightConditions: NightConditions;
  nightTitle: string;
}) {
  const completion = nightState.completionFraction;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="font-bold">{nightTitle}</h4>
      <div className="flex flex-row flex-wrap gap-2">
        <Badge className="text-sm">
          Faults time: {nightState.timeLoss.fault.toFixed(2)}
        </Badge>
        <Badge className="text-sm">
          Weather time: {nightState.timeLoss.weather.toFixed(2)}
        </Badge>
        <Badge className="text-sm">
          Unscheduled time: {nightState.timeLoss.unschedule.toFixed(2)}
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
