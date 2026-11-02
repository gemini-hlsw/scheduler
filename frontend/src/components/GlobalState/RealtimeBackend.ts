import { createContext } from "react";
import { ReactiveVar } from "@apollo/client";
import {
  isRealtimeConnectedVar,
  isSimulationConnectedVar,
} from "@/apollo-client";

// The RT-mode server the shared panels talk to. Operation is the default;
// the Simulation tab provides its own dyno so its runs stay apart.
export interface RealtimeBackend {
  clientName: string;
  name: string;
  isConnectedVar: ReactiveVar<boolean>;
}

export const OPERATION_BACKEND: RealtimeBackend = {
  clientName: "realtimeClient",
  name: "Operation",
  isConnectedVar: isRealtimeConnectedVar,
};

export const SIMULATION_BACKEND: RealtimeBackend = {
  clientName: "simulationClient",
  name: "Simulation",
  isConnectedVar: isSimulationConnectedVar,
};

export const RealtimeBackendContext =
  createContext<RealtimeBackend>(OPERATION_BACKEND);
