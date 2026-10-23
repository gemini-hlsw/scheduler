import { graphql } from "@/gql";

export const updateBuildParameters = graphql(`
  mutation updateBuildParams($buildParamsInput: BuildParametersInput!) {
    updateBuildParams(buildParamsInput: $buildParamsInput)
  }
`);

export const setProgramSelection = graphql(`
  mutation setProgramSelection($programIds: [String!]!, $selected: Boolean!) {
    setProgramSelection(programIds: $programIds, selected: $selected) {
      deselected
    }
  }
`);

export const resetProgramSelection = graphql(`
  mutation resetProgramSelection {
    resetProgramSelection {
      deselected
    }
  }
`);
