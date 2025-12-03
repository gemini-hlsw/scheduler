from asyncio import Queue
from typing import Dict

# Queue to manage scheduler process requests
scheduler_process_queue = Queue()

# Queue to manage manual plan trigger requests
plan_request_queue = Queue()

# Dictionary to hold queues for plan responses per schedule ID
plan_response_queue: Dict[str, Queue] = {}