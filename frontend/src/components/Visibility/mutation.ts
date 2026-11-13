import { graphql } from "@/gql";

/** Answers with the whole list after the change; every browser also gets it pushed. */
export const setObservationsIgnoredMutation = graphql(`
  mutation setObservationsIgnored(
    $observationIds: [String!]!
    $ignored: Boolean!
  ) {
    setObservationsIgnored(observationIds: $observationIds, ignored: $ignored) {
      observationId
      ignoredAt
    }
  }
`);
