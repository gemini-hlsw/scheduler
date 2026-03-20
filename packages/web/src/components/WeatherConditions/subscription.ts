import { graphql } from "@/../../schema/web";

export const weatherUpdatesSubscription = graphql(`
  subscription weatherUpdates {
    weatherUpdates {
      site
      imageQuality
      cloudCover
      windDirection
      windSpeed
    }
  }
`);
