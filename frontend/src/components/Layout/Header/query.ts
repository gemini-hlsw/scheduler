import { graphql } from "@/gql";
export const scheduleVersionQuery = graphql(`
  query version {
    version {
      version
      changelog
    }
  }
`);
