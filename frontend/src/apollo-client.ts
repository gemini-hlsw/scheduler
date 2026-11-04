import {
  ApolloClient,
  InMemoryCache,
  makeVar,
  ReactiveVar,
} from "@apollo/client";
import { split, HttpLink } from "@apollo/client";
import { getMainDefinition } from "@apollo/client/utilities";
import { createClient } from "graphql-ws";
import { GraphQLWsLink } from "@apollo/client/link/subscriptions";

const SCHEDULER_BACKEND_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000/graphql";

const REALTIME_BACKEND_URL =
  import.meta.env.VITE_REALTIME_API_URL ?? "http://localhost:8000/graphql";

// Same RT mode as the operation server, on its own dyno, so a simulation never
// touches the operation night.
const SIMULATION_BACKEND_URL =
  import.meta.env.VITE_SIMULATION_API_URL ?? "http://localhost:8000/graphql";

const WEATHER_BACKEND_URL =
  import.meta.env.VITE_WEATHER_URL ?? "http://localhost:4000";

// The weather service the simulation dyno listens to, so simulated weather
// never reaches the operation one.
const SIMULATION_WEATHER_BACKEND_URL =
  import.meta.env.VITE_SIMULATION_WEATHER_URL ?? "http://localhost:4000";

const SCHEDULER_API_URL = new URL(SCHEDULER_BACKEND_URL);
const REALTIME_API_URL = new URL(REALTIME_BACKEND_URL);
const SIMULATION_API_URL = new URL(SIMULATION_BACKEND_URL);
const WEATHER_API_URL = new URL(WEATHER_BACKEND_URL);
const SIMULATION_WEATHER_API_URL = new URL(SIMULATION_WEATHER_BACKEND_URL);

function createWsLink(
  url: URL,
  clientName: string,
  connectedVar?: ReactiveVar<boolean>
) {
  return new GraphQLWsLink(
    createClient({
      url: `${url.protocol === "https:" ? "wss" : "ws"}://${url.host}${
        url.pathname
      }`,
      keepAlive: 10000,
      retryAttempts: Infinity,
      shouldRetry: () => true,
      on: {
        connected: () => {
          if (connectedVar) {
            connectedVar(true);
          }
          console.log(
            `${clientName} Subscription connected successfully to ${url}`
          );
        },
        error: (error) => {
          if (connectedVar) {
            connectedVar(false);
          }
          console.log(
            `${clientName} Error connecting to subscription server ${url}`,
            error
          );
        },
        closed: () => {
          if (connectedVar) {
            connectedVar(false);
          }
          console.log(`${clientName} Subscription connection closed to ${url}`);
        },
      },
    })
  );
}

export const isWeatherConnectedVar = makeVar(false);
export const isSimulationWeatherConnectedVar = makeVar(false);
export const isSchedulerConnectedVar = makeVar(false);
export const isRealtimeConnectedVar = makeVar(false);
export const isSimulationConnectedVar = makeVar(false);

const wsWeatherLink = createWsLink(
  WEATHER_API_URL,
  "Weather",
  isWeatherConnectedVar
);
const wsSimulationWeatherLink = createWsLink(
  SIMULATION_WEATHER_API_URL,
  "Simulation Weather",
  isSimulationWeatherConnectedVar
);
const wsSchedulerLink = createWsLink(
  SCHEDULER_API_URL,
  "Scheduler",
  isSchedulerConnectedVar
);
const wsRealtimeLink = createWsLink(
  REALTIME_API_URL,
  "Realtime",
  isRealtimeConnectedVar
);
const wsSimulationLink = createWsLink(
  SIMULATION_API_URL,
  "Simulation",
  isSimulationConnectedVar
);

const httpSchedulerLink = new HttpLink({
  uri: SCHEDULER_BACKEND_URL,
});

const httpRealtimeLink = new HttpLink({
  uri: REALTIME_BACKEND_URL,
});

const httpSimulationLink = new HttpLink({
  uri: SIMULATION_BACKEND_URL,
});

const httpWeatherLink = new HttpLink({
  uri: WEATHER_BACKEND_URL,
});

const httpSimulationWeatherLink = new HttpLink({
  uri: SIMULATION_WEATHER_BACKEND_URL,
});

const splitLink = split(
  (operation) => {
    const definition = getMainDefinition(operation.query);
    return (
      definition.kind === "OperationDefinition" &&
      definition.operation === "subscription"
    );
  },
  split(
    (op) => op.getContext().clientName === "weatherClient",
    wsWeatherLink,
    split(
      (op) => op.getContext().clientName === "realtimeClient",
      wsRealtimeLink,
      split(
        (op) => op.getContext().clientName === "simulationClient",
        wsSimulationLink,
        split(
          (op) => op.getContext().clientName === "simulationWeatherClient",
          wsSimulationWeatherLink,
          wsSchedulerLink
        )
      )
    )
  ),
  split(
    (op) => op.getContext().clientName === "weatherClient",
    httpWeatherLink,
    split(
      (op) => op.getContext().clientName === "realtimeClient",
      httpRealtimeLink,
      split(
        (op) => op.getContext().clientName === "simulationClient",
        httpSimulationLink,
        split(
          (op) => op.getContext().clientName === "simulationWeatherClient",
          httpSimulationWeatherLink,
          httpSchedulerLink
        )
      )
    )
  )
);

export const client = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
});
