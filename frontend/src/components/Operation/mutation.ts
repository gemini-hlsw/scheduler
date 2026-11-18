import { graphql } from "@/gql";

// Accept or discard the proposed plan for every UI. The outcome arrives on the
// queueSchedule subscription as a PlanDecision, to this browser as well.
export const decidePlanMutation = graphql(`
  mutation decidePlan($proposalId: String!, $accept: Boolean!) {
    decidePlan(proposalId: $proposalId, accept: $accept) {
      proposalId
      accepted
      decidedBy
    }
  }
`);
