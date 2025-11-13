import asyncio

import gpp_client

from scheduler.night_monitor import EventListener, EventConsumer


class NightMonitor:


    def __init__(self, night):
        # The shared queue for events
        self.event_queue = asyncio.Queue()
        client = None
        self.listener = EventListener(client, self.event_queue)
        self.consumer = EventConsumer(self.event_queue)

        self._listener_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None

        self._shutdown_event = asyncio.Event()


    async def start(self):

        self._listener_task = asyncio.create_task(self.listener.listen())
        self._consumer_task = asyncio.create_task(self.consumer.consume())


    async def shutdown(self, drain_queue: bool = True):
        print("Shutting down the Night Monitor.")
        self._shutdown_event.set()


        if self._listener_task:
            self._listener_task.cancel()

        tasks = []
        if self._listener_task:
            tasks.append(self._listener_task)
        if self._consumer_task:
            tasks.append(self._consumer_task)

        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=1.0)

            # Force cancel any still-pending tasks
            for task in pending:
                print(f"Force cancelling pending task: {task.get_name()}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if drain_queue and not self.event_queue.empty():
            try:
                await asyncio.wait_for(self.event_queue.join(), timeout=2.0)
            except asyncio.TimeoutError:
                print(f"Queue drain timed out, {self.event_queue.qsize()} items remaining")

        print("Shutdown complete")