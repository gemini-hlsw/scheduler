import { graphql } from "@/gql";
export const scheduleV2Query = graphql(`
  query scheduleV2 {
    scheduleV2
  }
`);

export const operationNightQuery = graphql(`
  query operationNight {
    operationNight {
      site
      start
      end
    }
  }
`);

export const onDemandQuery = graphql(`
  query onDemandQuery {
    onDemandSchedule
  }
`);
