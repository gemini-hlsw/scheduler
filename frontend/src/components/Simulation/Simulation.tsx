import { useContext } from "react";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import {
  RealtimeBackendContext,
  SIMULATION_BACKEND,
} from "../GlobalState/RealtimeBackend";
import WeatherConditions from "../WeatherConditions/WeatherConditions";
import { DisplayWeather } from "../WeatherConditions/DisplayWeather";
import Result from "../Operation/Result";
import { cn } from "@/lib/utils";
import OnDemandControl from "../ControlPanel/OnDemandControl";
import BuildParameters from "../BuildParameters/BuildParameters";

// Same panels as Operation, plus what only makes sense off the real night:
// replaying it from evening twilight and picking another visibility range.
// They talk to the simulation dyno.
export function Simulation() {
  const { simNightPlans } = useContext(GlobalStateContext);

  return (
    <RealtimeBackendContext.Provider value={SIMULATION_BACKEND}>
      <div className="flex flex-col gap-2">
        <OnDemandControl />
        <div className="flex flex-row md:flex-row w-full gap-2">
          <div className={cn("grow flex gap-2 flex-row")}>
            <BuildParameters vertical={true} />
            <WeatherConditions vertical={true} updateButton={true} />
            <DisplayWeather />
          </div>
        </div>
        <Result plans={simNightPlans} />
      </div>
    </RealtimeBackendContext.Provider>
  );
}
