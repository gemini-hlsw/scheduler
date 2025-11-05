import asyncio

from scheduler.night_monitor import EventSourceType
from scheduler.night_monitor.event_handler import (
    EventHandler, ResourceEventHandler, WeatherEventHandler,
    ODBEventHandler
)

__all__ = ['EventConsumer']

class EventConsumer:
    """
    Consumes the events retrieved by the Listener from the queue
    so it can be handled by the corresponding Handler.

    Args:
        queue (asyncio.Queue): Queue to receive events from.
    """

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.resource_handler = ResourceEventHandler()
        self.weather_handler = WeatherEventHandler()
        self.odb_handler = ODBEventHandler()


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
        while True:
           try:
               item = await self.queue.get()

               if item is None:
                   # Poison pill
                   raise RuntimeError('Corrupt message was received')

               source, data = item
               handler = self._match_source_to_handler(source)

               try:
                   event = handler.parse_event(data)
                   await handler.handle(event)
               except Exception as e:
                   print("Handling process failed:", e)
               finally:
                   self.queue.task_done()


           except asyncio.CancelledError:
               print('Consumer cancelled')
           except RuntimeError:
               print('Consumer error')
               break