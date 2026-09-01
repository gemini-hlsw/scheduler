import { graphql } from "@/gql";

export const getProgramList = graphql(`
  query availablePrograms($nightDate: Date) {
    availablePrograms(nightDate: $nightDate) {
      id
      refLabel
    }
  }
`);

export const buildParametersQuery = graphql(`
  query buildParameters {
    buildParameters {
      nightTimes {
        site
        start
        end
      }
      visibilityStart
      visibilityEnd
      programList
      simulatedNow
    }
  }
`);
