# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import inspect
from typing import Callable

from scheduler.graphql_mid.types import NewPlansRT
from scheduler.services import logger_factory
from scheduler.core.events.queue import Event

__all__ = ["SchedulerQueue"]

_logger = logger_factory.create_logger(__name__)

class SchedulerQueue:

    def __init__(self):
        self._queue = asyncio.Queue()

    async def add_schedule_event(self, event: Event):
        """
        Publishes a scheduler event to the queue.

        Args:
            event: The event that triggered the scheduler event.
        """
        if event is None:
            _logger.error("Attempted to add a None event to the scheduler queue. Event will be ignored.")
            return
        self._queue.put_nowait(event)
        _logger.info(f"Sent Scheduler event: {event.description} ")

    async def consume_events(self, callback: Callable[[Event], NewPlansRT]):
        """
        Starts consuming events from the queue.

        Args:
            callback: Async function to process each event
        """

        event = await self._queue.get()
        try:
            _logger.info(f"Received Scheduler event: {event.description} at {event.time} ")

            # Call the user's callback
            result = None
            if callback is not None:
                if inspect.iscoroutinefunction(callback):
                    result = await callback(event)
                else:
                    result = callback(event)
                _logger.info("Callback executed successfully")
            else:
                raise ValueError('No callback function provided')
            
            return event, result

        except Exception as e:
            _logger.error(f"Error trying to consume event {event.description}: {e}")
            raise

    async def close(self):
        """
        Closes the connection gracefully.
        """
        while not self._queue.empty():
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break


