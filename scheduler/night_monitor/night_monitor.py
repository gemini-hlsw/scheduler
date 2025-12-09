# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from astropy.time import Time
from typing import List

from lucupy.minimodel.site import Site

from scheduler.clients.scheduler_queue_client import SchedulerQueueClient
from scheduler.night_monitor import EventListener, EventConsumer
from scheduler.night_monitor.night_tracker import NightTracker

from scheduler.services import logger_factory
_logger = logger_factory.create_logger(__name__)

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
    def __init__(self, night: Time, sites: List[Site]):
        # The shared queue for events
        self.event_queue = asyncio.Queue()
        self.scheduler_queue = SchedulerQueueClient()

        self._shutdown_event = asyncio.Event()

        client = None
        self.listener = EventListener(client, self.event_queue, self._shutdown_event)
        self.consumer = EventConsumer(self.event_queue, self._shutdown_event)
        self.night_tracker = NightTracker(night, sites)

        self._listener_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None
        self._night_tracker_task: asyncio.Task | None = None

    async def start(self):
        """
        Start the tasks for each subcomponent in the Night Monitor.
        """
        self._listener_task = asyncio.create_task(self.listener.listen())
        self._consumer_task = asyncio.create_task(self.consumer.consume())
        self._night_tracker_task = asyncio.create_task(self.night_tracker.start_tracking())


    async def shutdown(self, drain_queue: bool = True):
        """Signals the shutdown of the subcomponents and the Night monitor itself.

        Args:
            drain_queue (bool): If True, the event queue is drained. Defaults to True.
        """
        _logger.info("Shutting down the Night Monitor.")
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
                _logger.info(f"Force cancelling pending task: {task.get_name()}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    _logger.error(f"Task {task.get_name()} cancelled during shutdown.")
                    pass
        # drain the event_queue
        if drain_queue and not self.event_queue.empty():
            try:
                await asyncio.wait_for(self.event_queue.join(), timeout=2.0)
            except asyncio.TimeoutError:
                _logger.warning(f"Queue drain timed out, {self.event_queue.qsize()} items remaining")

        _logger.info("Shutdown complete")


    async def on_demand(self, params):
        """
        When an on-demand schedule happens we add to the queue with priority
        and reset other events.
        """
        scheduler_queue = SchedulerQueueClient()
        # TODO: add proper event to the schedule queue
        scheduler_queue.add_schedule_event()