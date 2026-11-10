import { createContext } from "react";
import { ReactiveVar } from "@apollo/client";
import {
  isRealtimeConnectedVar,
  isSimulationConnectedVar,
  isSimulationWeatherConnectedVar,
  isWeatherConnectedVar,
} from "@/apollo-client";

// The RT-mode server, and the weather service it listens to, that the shared
// panels talk to. Operation is the default; the Simulation tab provides its
// own pair so its runs stay apart.
//
// Gotcha: Apollo merges subscriptions with the same query and variables
// without looking at clientName, so on a tab switch the new panel would join
// the old tab's stream and never open its own socket (badge stuck offline,
// other backend's data). Subscriptions sent to both backends pass
// queryDeduplication: false in their context.
export interface RealtimeBackend {
  clientName: string;
  name: string;
  // The tab these panels sit on. Also the id of that tab's toaster in App, so
  // a toast raised by a shared panel only shows on its own tab.
  tab: "operation" | "simulation";
  isConnectedVar: ReactiveVar<boolean>;
  weatherClientName: string;
  isWeatherConnectedVar: ReactiveVar<boolean>;
}

export const OPERATION_BACKEND: RealtimeBackend = {
  clientName: "realtimeClient",
  name: "Operation",
  tab: "operation",
  isConnectedVar: isRealtimeConnectedVar,
  weatherClientName: "weatherClient",
  isWeatherConnectedVar: isWeatherConnectedVar,
};

export const SIMULATION_BACKEND: RealtimeBackend = {
  clientName: "simulationClient",
  name: "Simulation",
  tab: "simulation",
  isConnectedVar: isSimulationConnectedVar,
  weatherClientName: "simulationWeatherClient",
  isWeatherConnectedVar: isSimulationWeatherConnectedVar,
};

export const RealtimeBackendContext =
  createContext<RealtimeBackend>(OPERATION_BACKEND);
