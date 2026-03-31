import { useContext } from "react";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import WeatherConditions from "../WeatherConditions/WeatherConditions";
import { Result } from "./Result";
import { cn } from "@/lib/utils";
import { useLazyQuery } from "@apollo/client";
import { toUtcIsoString } from "@/helpers/utcTime";
import { scheduleV2Query } from "./query";
import {
  PROGRAM_LIST_XT2,
  ProgramListType,
} from "../ControlPanel/ProgramSelection/ProgramList";
import { DisplayWeather } from "../WeatherConditions/DisplayWeather";
import OnDemandControl from "../ControlPanel/OnDemandControl";
import BuildParameters from "../BuildParameters/BuildParameters";

export default function Operation() {
  const [scheduleV2] = useLazyQuery(scheduleV2Query, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  const {
    rtPlan,
    thesis,
    power,
    metPower,
    whaPower,
    airPower,
    visPower,
    loadingPlan,
    setLoadingPlan,
    imageQuality,
    cloudCover,
    windDirection,
    windSpeed,
  } = useContext(GlobalStateContext);

  function runPlan(
    date: { from: Date; to?: Date },
    startTime: Date,
    endTime: Date,
    site: string,
    programs: ProgramListType[]
  ) {
    setLoadingPlan(true);
    const variables = {
      startTime: toUtcIsoString(date.from),
      endTime: toUtcIsoString(date.to),
      nightStartTime: toUtcIsoString(startTime),
      nightEndTime: toUtcIsoString(endTime),
      imageQuality: imageQuality,
      cloudCover: cloudCover,
      windSpeed: windSpeed,
      windDirection: windDirection,
      sites: site,
      thesisFactor: thesis,
      power: power,
      whaPower: whaPower,
      airPower: airPower,
      metPower: metPower,
      visPower: visPower,
      programs: programs.filter((p) => p.checked).map((p) => p.id),
    };
    scheduleV2({
      variables: variables,
    });
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-row md:flex-row w-full gap-2">
        <div className={cn("grow flex gap-2 flex-row")}>
          <OnDemandControl
            vertical={true}
            runPlan={runPlan}
            programList={PROGRAM_LIST_XT2}
            loadingPlan={loadingPlan}
          />
          <BuildParameters vertical={true} />
          <WeatherConditions vertical={true} updateButton={true} />
          <DisplayWeather />
        </div>
      </div>
      <Result rtPlan={rtPlan} />
    </div>
  );
}
