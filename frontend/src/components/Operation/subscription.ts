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

export const planStatusSubscription = graphql(`
  subscription planStatusUpdates {
    planStatusUpdates {
      state
      since
      eventDescription
      requestedByUser
      lastFinishedAt
      lastDurationSeconds
      lastError
    }
  }
`);
