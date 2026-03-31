import { graphql } from "@/gql";

export const getProgramList = graphql(`
  query availablePrograms {
    availablePrograms {
      id
      refLabel
    }
  }
`);
