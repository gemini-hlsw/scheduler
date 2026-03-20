import { graphql } from "@/../../schema/web";

export const getProgramList = graphql(`
  query availablePrograms {
    availablePrograms {
      id
      refLabel
    }
  }
`);
