import { graphql } from "@/gql";

export const schedulerEventsSubscription = graphql(`
  subscription schedulerEvents {
    schedulerEvents {
      id
      receivedAt
      eventType
      sites
      eventTime
      description
      triggeredPlan
    }
  }
`);
