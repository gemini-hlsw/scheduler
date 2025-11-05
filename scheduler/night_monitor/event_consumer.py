import asyncio


from scheduler.night_monitor import EventSourceType


class EventConsumer:


    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.resource_handler = None
        self.weather_handler = None
        self.odb_handler = None


    async def _match_source_to_handler(self, source: EventSourceType):

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
        while True:
           try:
               item = await self.queue.get()

               if item is None:
                   print('Poison pill')
                   break

               source, data = item
               handler = self._match_source_to_handler(source)


               try:
                   event = handler.parse_event(data)
                   await handler.handle(event)
               except Exception as e:
                   print(e)
               finally:
                   self.queue.task_done()


           except asyncio.CancelledError:
               print('Consumer cancelled')
           except RuntimeError:
               print('Consumer error')
