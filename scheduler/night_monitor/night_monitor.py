# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio

from scheduler.clients.scheduler_queue_client import SchedulerQueueClient
from scheduler.night_monitor import EventListener, EventConsumer

__all__ = ['NightMonitor']

class NightMonitor:
    """
    Tracks all the events that can happened throughout the night from different sources.

    Attributes:
        event_queue (asyncio.Queue): Queue to hold the events received from different subscriptions.
        scheduler_queue (SchedulerQueueClient): Client to interact with the Scheduler Queue that commands new schedules.
        listener (EventListener): Event listener handles all subscriptions connections and messages.
        consumer (EventConsumer): Event Consumer that receives serialized events from subscriptions.

        _shutdown_event (asyncio.Event): Signals shutting down for all subcomponents.
        _listener_task (asyncio.Task): Holds the Event listener process.
        _consumer_task (asyncio.Task): Holds the Event Consumer process.

    """
    def __init__(self, night):
        # The shared queue for events
        self.event_queue = asyncio.Queue()
        self.scheduler_queue = SchedulerQueueClient()

        self._shutdown_event = asyncio.Event()

        client = None
        self.listener = EventListener(client, self.event_queue, self._shutdown_event)
        self.consumer = EventConsumer(self.event_queue, self._shutdown_event)

        self._listener_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None
        # self._night_traker_task: asyncio.Task | None = None

    async def start(self):
        """
        Start the tasks for each subcomponent in the Night Monitor.
        """
        self._listener_task = asyncio.create_task(self.listener.listen())
        self._consumer_task = asyncio.create_task(self.consumer.consume())
        # self._night_traker ...


    async def shutdown(self, drain_queue: bool = True):
        """Signals the shutdown of the subcomponents and the Night monitor itself.

        Args:
            drain_queue (bool): If True, the event queue is drained. Defaults to True.
        """
        print("Shutting down the Night Monitor.")
        self._shutdown_event.set()

        # Clean listener
        if self._listener_task:
            self._listener_task.cancel()

        # Wait for the task to end briefly
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
        # drain the event_queue
        if drain_queue and not self.event_queue.empty():
            try:
                await asyncio.wait_for(self.event_queue.join(), timeout=2.0)
            except asyncio.TimeoutError:
                print(f"Queue drain timed out, {self.event_queue.qsize()} items remaining")

        print("Shutdown complete")


    async def on_demand(self, params):
        """
        When an on-demand schedule happens we add to the queue with priority
        and reset other events.
        """
        scheduler_queue = SchedulerQueueClient()
        scheduler_queue.add_schedule_event()