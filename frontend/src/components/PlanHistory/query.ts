import { graphql } from "@/gql";

/** Recent nights with stored plans, newest first, for the night picker. */
export const planNightsQuery = graphql(`
  query generatedPlanNights($mode: String) {
    generatedPlanNights(mode: $mode) {
      nightDate
      planCount
    }
  }
`);

/** A night's plans without their content: cheap enough to list them all. */
export const nightPlansQuery = graphql(`
  query generatedPlans($nightDate: Date!, $mode: String) {
    generatedPlans(nightDate: $nightDate, mode: $mode) {
      id
      createdAt
      mode
      appVersion
      nightDate
      eventType
      eventDescription
      eventTime
      eventSite
      status
      proposalId
      decidedBy
      decidedAt
    }
  }
`);

/** One plan, `{nightPlans, plansSummary}` as the UI received it when it was made. */
export const planContentQuery = graphql(`
  query generatedPlanContent($planId: Int!) {
    generatedPlanContent(planId: $planId)
  }
`);
