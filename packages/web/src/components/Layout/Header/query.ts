import { graphql } from "@/../../schema/web";
export const scheduleVersionQuery = graphql(`
  query version {
    version {
      version
      changelog
    }
  }
`);
