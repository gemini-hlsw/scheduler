from asyncio import Queue

# Dictionary to hold queues for plan responses per schedule ID
plan_response_subscribers: dict[str, set[Queue]] = {}
# The last plan published per schedule ID, replayed to every new subscriber so a
# reloaded UI gets it at once instead of waiting for the next plan. Only the
# operation process publishes here: validation runs use a new ID per page load.
last_plan_responses: dict[str, object] = {}
build_parameters_subscribers: set[Queue] = set()
program_selection_subscribers: set[Queue] = set()
ignored_observations_subscribers: set[Queue] = set()
