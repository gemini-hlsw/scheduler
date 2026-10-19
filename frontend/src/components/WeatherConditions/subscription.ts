import { graphql } from "@/gql";

export const weatherUpdatesSubscription = graphql(`
  subscription weatherUpdates {
    weatherUpdates {
      site
      imageQuality
      cloudCover
      waterVapor
      skyBackground
      windDirection
      windSpeed
    }
  }
`);
