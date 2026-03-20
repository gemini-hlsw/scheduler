import { graphql } from "@/../../schema/web";

export const updateBuildParameters = graphql(`
  mutation updateBuildParams($buildParamsInput: BuildParametersInput!) {
    updateBuildParams(buildParamsInput: $buildParamsInput)
  }
`);
