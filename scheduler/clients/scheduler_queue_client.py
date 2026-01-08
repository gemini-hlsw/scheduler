# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from datetime import datetime
from typing import Callable, Any

from lucupy.minimodel import Site
from pydantic import BaseModel, field_serializer, field_validator

from scheduler.graphql_mid.types import NewPlansRT
from scheduler.events import NightEvent
from scheduler.services import logger_factory

__all__ = ["SchedulerQueue", "SchedulerEvent"]

_logger = logger_factory.create_logger(__name__)

class SchedulerEvent(BaseModel):
    trigger_event: str
    time: datetime | None = None
    site: Site | None = None

    @field_serializer('site')
    def serialize_site(self, site: Site) -> str:
        """Convert Site enum to string"""
        return site.site_name

    @field_validator('site', mode='before')
    @classmethod
    def parse_site(cls, value):
        """Convert string to Site enum"""
        if isinstance(value, str):
            return Site.GN if value == 'Gemini North' else Site.GS
        return value

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
            _logger.info(f"Received Scheduler event: {event.trigger_event} ")

            # Call the user's callback
            if callback is not None:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                _logger.info("Callback executed successfully")
            else:
                raise ValueError('No callback function provided')

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


