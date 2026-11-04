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
export interface RealtimeBackend {
  clientName: string;
  name: string;
  isConnectedVar: ReactiveVar<boolean>;
  weatherClientName: string;
  isWeatherConnectedVar: ReactiveVar<boolean>;
}

export const OPERATION_BACKEND: RealtimeBackend = {
  clientName: "realtimeClient",
  name: "Operation",
  isConnectedVar: isRealtimeConnectedVar,
  weatherClientName: "weatherClient",
  isWeatherConnectedVar: isWeatherConnectedVar,
};

export const SIMULATION_BACKEND: RealtimeBackend = {
  clientName: "simulationClient",
  name: "Simulation",
  isConnectedVar: isSimulationConnectedVar,
  weatherClientName: "simulationWeatherClient",
  isWeatherConnectedVar: isSimulationWeatherConnectedVar,
};

export const RealtimeBackendContext =
  createContext<RealtimeBackend>(OPERATION_BACKEND);
