# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio

from scheduler.core.events.queue.nightly_timeline_store import NightlyTimelineStore
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue
from scheduler.night_monitor.event_sources import EventSourceType
from scheduler.night_monitor.event_handlers import (
    EventHandler, WeatherEventHandler,
    ODBEventHandler, ResourceEventHandler
)
from scheduler.services import logger_factory

__all__ = ['EventConsumer']

_logger = logger_factory.create_logger(__name__)

class EventConsumer:
    """
    Consumes the events retrieved by the Listener from the queue
    so it can be handled by the corresponding Handler.

    Args:
        event_queue (asyncio.Queue): Queue to receive events from that is shared with the Listener.
        shutdown_event (asyncio.Event): Event to stop consuming linked with the Listener.
        scheduler_queue (SchedulerQueue): Use to send new schedule request to the Engine.
        nightly_timeline_store (NightlyTimelineStore): Shared store the handlers read to know the
            plan currently in effect.
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        shutdown_event: asyncio.Event,
        scheduler_queue: SchedulerQueue,
        nightly_timeline_store: NightlyTimelineStore
    ):
        self.queue = event_queue
        self.scheduler_queue = scheduler_queue
        self.nightly_timeline_store = nightly_timeline_store
        self.resource_handler = ResourceEventHandler(self.scheduler_queue, nightly_timeline_store)
        self.weather_handler = WeatherEventHandler(self.scheduler_queue, nightly_timeline_store)
        self.odb_handler = ODBEventHandler(self.scheduler_queue, nightly_timeline_store)
        self._shutdown_event = shutdown_event


    def _match_source_to_handler(self, source: EventSourceType) -> EventHandler:
        """
        Matches the source to an event handler.

        Args:
            source (EventSourceType): Event source type specified in the item.

        Returns:
            EventHandler: Matched event handler.

        Raises:
            RuntimeError: If source is not matching the current set of handlers

        """
        match source:
            case EventSourceType.RESOURCE:
                return self.resource_handler
            case EventSourceType.WEATHER:
                return self.weather_handler
            case EventSourceType.ODB:
                return self.odb_handler
            case _:
                raise RuntimeError(f'Unknown event source: {source}')

    async def consume(self):
        """
        Consumes the events from the queue.

        One bad event (unknown source, corrupt item, handler bug) is logged
        and skipped: it must not kill the consumer for the rest of the night.
        """
        while not self._shutdown_event.is_set():
           try:
               item = await self.queue.get()
               try:
                   source, sub_name, data = item
                   handler = self._match_source_to_handler(source)
                   await handler.handle(sub_name, data)
               except asyncio.CancelledError:
                   raise
               except Exception:
                   _logger.exception(f'Failed to handle event {repr(item)}; skipping it.')
               finally:
                   self.queue.task_done()

           except asyncio.CancelledError:
               _logger.info('Event consumer cancelled.')
               break