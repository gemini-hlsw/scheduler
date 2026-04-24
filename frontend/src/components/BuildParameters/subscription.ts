import { graphql } from "@/gql";

export const buildParametersSubscription = graphql(`
  subscription buildParametersUpdates {
    buildParametersUpdates {
      nightTimes {
        site
        start
        end
      }
      visibilityStart
      visibilityEnd
      programList
    }
  }
`);
