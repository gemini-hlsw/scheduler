import { useMemo, useReducer } from "react";
import { NightPlanType } from "../../types";

/*
 * What a live (operation or simulation) backend says about its plans. The
 * server decides which plan is in effect; this only mirrors it:
 *
 * - `current` is the plan in effect (NewNightPlans): the night's first plan,
 *   or the last one accepted.
 * - `pending` is a plan computed on top of it and proposed
 *   (ProposedNightPlans), drawn below `current` so it can be compared.
 *
 * Accepting or discarding goes to the server (decidePlan), which tells every
 * UI with a PlanDecision; only then does this state change, so all browsers
 * show the same plan. A reloaded page gets the plan in effect, then the
 * proposal if one is still waiting.
 */

export interface PendingPlan {
  proposalId: string;
  plans: NightPlanType[];
}

export interface PlanReview {
  current: NightPlanType[];
  /** The newest proposed plan, or null when none is waiting. */
  pending: PendingPlan | null;
  /** The plan in effect, from the server. */
  inEffect: (plans: NightPlanType[]) => void;
  /** A plan proposed by the server; replaces an older proposal. */
  proposed: (proposalId: string, plans: NightPlanType[]) => void;
  /** The server settled a proposal, maybe from another browser. */
  decided: (proposalId: string, accepted: boolean) => void;
  /**
   * The backend failed to make a plan, or the proposal could not be decided
   * any more: drop the pending one, keep current.
   */
  dropPending: () => void;
}

interface State {
  current: NightPlanType[];
  pending: PendingPlan | null;
}

type Action =
  | { type: "inEffect"; plans: NightPlanType[] }
  | { type: "proposed"; pending: PendingPlan }
  | { type: "decided"; proposalId: string; accepted: boolean }
  | { type: "dropPending" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "inEffect":
      // The server only sends a plan in effect when no proposal is waiting.
      return { current: action.plans, pending: null };
    case "proposed":
      return { ...state, pending: action.pending };
    case "decided":
      // A decision on an older proposal than the one shown changes nothing.
      if (state.pending?.proposalId !== action.proposalId) return state;
      return action.accepted
        ? { current: state.pending.plans, pending: null }
        : { ...state, pending: null };
    case "dropPending":
      return state.pending === null ? state : { ...state, pending: null };
  }
}

export function usePlanReview(): PlanReview {
  const [state, dispatch] = useReducer(reducer, {
    current: [],
    pending: null,
  });

  return useMemo(
    () => ({
      ...state,
      inEffect: (plans) => dispatch({ type: "inEffect", plans }),
      proposed: (proposalId, plans) =>
        dispatch({ type: "proposed", pending: { proposalId, plans } }),
      decided: (proposalId, accepted) =>
        dispatch({ type: "decided", proposalId, accepted }),
      dropPending: () => dispatch({ type: "dropPending" }),
    }),
    [state],
  );
}
