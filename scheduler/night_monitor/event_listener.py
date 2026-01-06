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

from ..clients import SchedulerQueue


class SubscriptionEndedException(Exception): pass
RETRYABLE_EXCEPTIONS = (
    ConnectionError, asyncio.TimeoutError,
    ClientError, ConnectionClosedError, InvalidStatus,
)


class EventListener:
    """
    Handles all subscriptions that generates events and store them so they can be retrieved from the EventConsumer.
    """
    def __init__(
        self,
        client,
        queue: asyncio.Queue,
        shutdown_event: asyncio.Event
    ):
        self.queue = queue
        self._sources = [
            ResourceEventSource(client),
            WeatherEventSource(client),
            ODBEventSource(client)
        ]
        self._shutdown_event = shutdown_event

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
        try:
            sub_generator = await subscription_factory()

            async for data in sub_generator:
                if self._shutdown_event.is_set():
                    break
                await self.queue.put((source, sub_name, data))

            if not self._shutdown_event.is_set():
                raise SubscriptionEndedException(f"Subscription '{sub_name}' ended gracefully, retrying.")

        except asyncio.CancelledError:
            raise



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
       try:
           await asyncio.gather(*producer_tasks, return_exceptions=True)
       except asyncio.CancelledError:
           for task in producer_tasks:
               if not task.done():
                   task.cancel()
