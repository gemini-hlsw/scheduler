# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio

from scheduler.clients import SchedulerQueue
from scheduler.night_monitor.event_sources import EventSourceType
from scheduler.night_monitor.event_handlers import (
    EventHandler, WeatherEventHandler,
    ODBEventHandler, ResourceEventHandler
)

__all__ = ['EventConsumer']

class EventConsumer:
    """
    Consumes the events retrieved by the Listener from the queue
    so it can be handled by the corresponding Handler.

    Args:
        event_queue (asyncio.Queue): Queue to receive events from that is shared with the Listener.
        shutdown_event (asyncio.Event): Event to stop consuming linked with the Listener.
        scheduler_queue (SchedulerQueue): Use to send new schedule request to the Engine.
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        shutdown_event: asyncio.Event,
        scheduler_queue: SchedulerQueue
    ):
        self.queue = event_queue
        self.scheduler_queue = scheduler_queue
        self.resource_handler = ResourceEventHandler(self.scheduler_queue)
        self.weather_handler = WeatherEventHandler(self.scheduler_queue)
        self.odb_handler = ODBEventHandler(self.scheduler_queue)
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

        Raises:
            RuntimeError: a returned item from the queue is invalid.
        """
        while not self._shutdown_event.is_set():
           try:
               item = await self.queue.get()

               # if item is None:
               #    # Poison pill
               #    raise RuntimeError('Corrupt message was received')

               source, sub_name, data = item
               handler = self._match_source_to_handler(source)

               try:
                   await handler.handle(sub_name, data)
               except Exception as e:
                   print("Handling process failed:", e)
               finally:
                   self.queue.task_done()


           except asyncio.CancelledError:
               print('Consumer cancelled')
               break
           except RuntimeError:
               print('Consumer error')
               break