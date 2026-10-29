import { useContext } from "react";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import WeatherConditions from "../WeatherConditions/WeatherConditions";
import Result from "./Result";
import { cn } from "@/lib/utils";
import { DisplayWeather } from "../WeatherConditions/DisplayWeather";
import OnDemandControl from "../ControlPanel/OnDemandControl";
import BuildParameters from "../BuildParameters/BuildParameters";
import SchedulerEvents from "./SchedulerEvents";

export default function Operation() {
  const { nightPlans } = useContext(GlobalStateContext);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-row md:flex-row w-full gap-2">
        <div className={cn("grow flex gap-2 flex-row")}>
          <BuildParameters
            vertical={true}
            showVisibility={false}
            buttonsRow={true}
            operationNightTimes={true}
          />
          {/* The run button sits over the two weather panels, whose column
              is shorter than the build parameters, so no panel has to
              stretch far to match the others. */}
          <div className="flex flex-col gap-2">
            <OnDemandControl showStartOfNight={false} />
            <div className="flex flex-row gap-2 grow">
              <WeatherConditions vertical={true} updateButton={true} />
              <DisplayWeather />
            </div>
          </div>
        </div>
      </div>
      <SchedulerEvents />
      <Result plans={nightPlans} />
    </div>
  );
}
