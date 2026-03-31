import { HttpLink } from "@apollo/client";
import { createClient } from "graphql-ws";
import { GraphQLWsLink } from "@apollo/client/link/subscriptions";

export function createApolloClient(url: string | undefined) {
  if (!url) {
    throw new Error("No url provided");
  }

  const API_URL = new URL(url);

  const wsLink = new GraphQLWsLink(
    createClient({
      url: `${API_URL.protocol === "https:" ? "wss" : "ws"}://${API_URL.host}${
        API_URL.pathname
      }`,
      keepAlive: 10000,
      retryAttempts: Infinity,
      shouldRetry: () => true,
      on: {
        connected: () => {
          console.log(`Subscription connected successfully to ${url}`);
        },
        error: (error) => {
          console.log(`Error connecting to subscription server ${url}`, error);
        },
        closed: () => {
          console.log(`Subscription connection closed to ${url}`);
        },
      },
    })
  );

  const httpLink = new HttpLink({
    uri: url,
  });

  return { wsLink, httpLink };
}
