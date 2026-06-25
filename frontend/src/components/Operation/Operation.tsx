import { useContext } from "react";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import WeatherConditions from "../WeatherConditions/WeatherConditions";
import { Result } from "./Result";
import { cn } from "@/lib/utils";
import { DisplayWeather } from "../WeatherConditions/DisplayWeather";
import OnDemandControl from "../ControlPanel/OnDemandControl";
import BuildParameters from "../BuildParameters/BuildParameters";

export default function Operation() {
  const { rtPlan } = useContext(GlobalStateContext);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-row md:flex-row w-full gap-2">
        <div className={cn("grow flex gap-2 flex-row")}>
          <OnDemandControl />
          <BuildParameters vertical={true} />
          <WeatherConditions vertical={true} updateButton={true} />
          <DisplayWeather />
        </div>
      </div>
      <Result rtPlan={rtPlan} />
    </div>
  );
}
