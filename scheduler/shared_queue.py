from asyncio import Queue
from typing import Dict

scheduler_process_queue = Queue()
plan_request_queue = Queue()
plan_response_queue: Dict[str, Queue] = {}