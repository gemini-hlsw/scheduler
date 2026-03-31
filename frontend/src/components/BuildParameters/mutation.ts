import { graphql } from "@/gql";

export const updateBuildParameters = graphql(`
  mutation updateBuildParams($buildParamsInput: BuildParametersInput!) {
    updateBuildParams(buildParamsInput: $buildParamsInput)
  }
`);
