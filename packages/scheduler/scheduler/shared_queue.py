from asyncio import Queue

# Dictionary to hold queues for plan responses per schedule ID
plan_response_subscribers: dict[str, set[Queue]] = {}
