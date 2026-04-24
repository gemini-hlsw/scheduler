import { graphql } from "@/gql";

export const getProgramList = graphql(`
  query availablePrograms {
    availablePrograms {
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
    }
  }
`);
