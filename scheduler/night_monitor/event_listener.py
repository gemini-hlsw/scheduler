# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import stamina
from aiohttp import ClientError
from websockets import ConnectionClosedError, InvalidStatus

from .event_sources import (
    ResourceEventSource,
    WeatherEventSource,
    ODBEventSource,
    EventSourceType,
)


__all__ = ['EventListener', 'SubscriptionEndedException']

class SubscriptionEndedException(Exception): pass
RETRYABLE_EXCEPTIONS = (
    ConnectionError, asyncio.TimeoutError,
    ClientError, ConnectionClosedError, InvalidStatus,
)


class EventListener:
    """
    Handles all subscriptions that generates events and store them so they can be retrieved from the EventConsumer.
    """
    def __init__(self, client):
        self.queue = asyncio.Queue()
        self._sources = [
            ResourceEventSource(client),
            WeatherEventSource(client),
            ODBEventSource(client)
        ]
        # self._shutdown_event = asyncio.Event()

    @stamina.retry(
        on=RETRYABLE_EXCEPTIONS,
        wait_initial=1.0,
        wait_max=10.0,
    )
    async def _producer(
            self,
            source: EventSourceType,
            sub_name: str,
            subscription_factory: callable
    ):
        """
        Calls the factory from each source and put the data on the queue.

        source (EventSourceType): Source of the subscription.
        sub_name (str): Name of the subscription called.
        subscription_factory (callable): Callable that returns the async generator that is used to retrieve the data.
        """
        sub_generator = await subscription_factory()

        async for data in sub_generator:
            await self.queue.put((source, sub_name, data))


    async def listen(self):
       """
       Starts and gathers all producer tasks.
       """
       producer_tasks = [
           asyncio.create_task(
               self._producer(
                   source.source_type,
                   sub_name,
                   sub
               )
           ) for source in self._sources for sub_name, sub in source.subscriptions()
       ]

       await asyncio.gather(*producer_tasks, return_exceptions=True)
