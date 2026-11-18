A new plan from the Operation or Simulation server is shown below the current plan, to be accepted or discarded, instead of replacing it. The night's first plan still takes effect at once.
While comparing, both plots are shorter, their observation tables are collapsed and the page scrolls to the new plan.
The decision is made on the server and applies to every connected user, who are told who decided; a reloaded page shows the plan in effect and the new plan still waiting.
Add the `decidePlan` mutation, and the `ProposedNightPlans` and `PlanDecision` results to the `queueSchedule` subscription.
