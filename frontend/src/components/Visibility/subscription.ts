import { graphql } from "@/gql";

/** The shared ignore list: the current one on connect, then every change. */
export const ignoredObservationsSubscription = graphql(`
  subscription ignoredObservationsUpdates {
    ignoredObservationsUpdates {
      observationId
      ignoredAt
    }
  }
`);
