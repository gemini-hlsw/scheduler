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
      visibilityNights
      programList
      simulatedNow
    }
  }
`);

export const programSelectionSubscription = graphql(`
  subscription programSelectionUpdates {
    programSelectionUpdates {
      deselected
    }
  }
`);
