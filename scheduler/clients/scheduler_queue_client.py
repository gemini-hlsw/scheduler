# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any

from lucupy.minimodel import Site, ALL_SITES
from pydantic import BaseModel, field_serializer, field_validator

from scheduler.graphql_mid.types import NewPlansRT
from scheduler.events import NightEvent, OnDemandScheduleEvent
from scheduler.services import logger_factory
from scheduler.core.events.queue import WeatherChangeEvent

__all__ = ["SchedulerQueue", "SchedulerEvent"]

_logger = logger_factory.create_logger(__name__)

@dataclass
class SchedulerEvent:
    trigger_event: str
    event: Any | None = None
    time: datetime | None = None
    site: Site | None = None

class SchedulerQueue:

    def __init__(self):
        self._queue = asyncio.Queue()

    async def add_schedule_event(
        self,
        reason: str,
        event: Any | None, # The type should be a collection of different events, a
        priority: int = 0):
        """
        Publishes a scheduler event to the queue.

        Args:
            reason (str): The reason for the event.
            event: The event that triggered the scheduler event.
            priority: Priority level (0 = regular, 1 = on_demand)
        """
        if event is not None:
            if isinstance(event, NightEvent):
                scheduler_event = SchedulerEvent(
                    trigger_event=reason,
                    time=event.time.to_datetime(),
                    site=event.site,
                )
            elif isinstance(event, OnDemandScheduleEvent):
                scheduler_event = SchedulerEvent(
                    trigger_event=reason,
                    time=event.time,
                    site=None
                )
            elif isinstance(event, WeatherChangeEvent):
                # TODO: Variant snapshot should be used to update weather before plan calculation
                # Maybe the entire event should be passed to get the variants in the consume_events function
                scheduler_event = SchedulerEvent(
                    trigger_event=reason,
                    event=event,
                    time=event.time,
                    site=event.site,
                )
            else:
                scheduler_event = SchedulerEvent(
                    trigger_event=reason,
                    time=event.time,
                    site=event.observation.site,
                )
        else:
            scheduler_event = SchedulerEvent(trigger_event=reason)
        self._queue.put_nowait(scheduler_event)
        _logger.info(f"Sent Scheduler event: {scheduler_event.trigger_event} ")

    async def consume_events(self, callback: Callable[[SchedulerEvent], NewPlansRT]):
        """
        Starts consuming events from the queue.

        Args:
            callback: Async function to process each event
        """

        try:
            event = await self._queue.get()
            _logger.info(f"Received Scheduler event: {event.trigger_event} at {event.time} ")

            # Call the user's callback
            result = None
            if callback is not None:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(event)
                else:
                    result = callback(event)
                _logger.info("Callback executed successfully")
            else:
                raise ValueError('No callback function provided')
            
            return event, result

        except Exception as e:
            print(f"Error trying to consume event {event.trigger_event}: {e}")
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


