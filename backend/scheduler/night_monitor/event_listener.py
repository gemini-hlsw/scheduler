# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from functools import partial

import stamina
from aiohttp import ClientError
from websockets import ConnectionClosedError, ConnectionClosedOK, InvalidStatus
from typing import Any

from .event_sources import (
    ResourceEventSource,
    WeatherEventSource,
    ODBEventSource,
    EventSourceType,
)

from scheduler.services import logger_factory
_logger = logger_factory.create_logger(__name__)


__all__ = ['EventListener', 'SubscriptionEndedException']


class SubscriptionEndedException(Exception): pass
# SubscriptionEndedException is raised below when a subscription ends without an
# error, which is what a server-side `complete` frame looks like from here: the
# gpp-client closes the socket and the async-for simply runs out. That and a
# clean 1000 close (ConnectionClosedOK) are ordinary events for a long-lived
# subscription, so both must reconnect rather than end the night.
RETRYABLE_EXCEPTIONS = (
    ConnectionError, asyncio.TimeoutError,
    ClientError, ConnectionClosedError, ConnectionClosedOK, InvalidStatus,
    SubscriptionEndedException,
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

    # attempts/timeout are None on purpose. stamina's defaults (10 attempts,
    # 45s) measure from when the call first started, not from the first failure,
    # so a subscription that has been streaming for longer than 45s gets zero
    # reconnect attempts: the first dropped socket ends it for the rest of the
    # night. A subscription must keep reconnecting for as long as the night
    # monitor is up; wait_max caps the backoff so an ODB that is down for hours
    # is retried every 10s rather than spun on.
    @stamina.retry(
        on=RETRYABLE_EXCEPTIONS,
        attempts=None,
        timeout=None,
        wait_initial=1.0,
        wait_max=10.0,
    )
    async def _producer(
            self,
            source: EventSourceType,
            sub_name: str,
            subscription_factory: callable,
            client: Any
    ):
        """
        Calls the factory from each source and put the data on the queue.

        source (EventSourceType): Source of the subscription.
        sub_name (str): Name of the subscription called.
        subscription_factory (callable): Callable that returns the async generator that is used to retrieve the data.
        """
        try:
            # Create the actual session
            if source == EventSourceType.WEATHER:
                if client is None:
                    raise ValueError("Client is not initialized for WeatherEventSource.")
                async with client as session:
                    sub_generator = subscription_factory(session)
                    async for data in sub_generator:
                        _logger.debug("Received Weather event:")
                        _logger.debug(data)
                        if self._shutdown_event.is_set():
                            break
                        await self.queue.put((source, sub_name, data))

                if not self._shutdown_event.is_set():
                    raise SubscriptionEndedException(f"Subscription '{sub_name}' ended gracefully, retrying.")

            else:
                _logger.info(f"Listening to {sub_name}")
                async for data in subscription_factory(client):
                    if self._shutdown_event.is_set():
                        break
                    await self.queue.put((source, sub_name, data))

                if not self._shutdown_event.is_set():
                    raise SubscriptionEndedException(f"Subscription '{sub_name}' ended gracefully, retrying.")

        except ValueError as e:
            raise e

        except asyncio.CancelledError:
            raise

        except RETRYABLE_EXCEPTIONS as e:
            # Retries are now unbounded, so a subscription that can never connect
            # would reconnect silently forever. Say so on every attempt: this is
            # the only trace left of a subscription that is up but not delivering.
            _logger.warning(
                f"Subscription '{sub_name}' dropped ({type(e).__name__}: {e}); reconnecting."
            )
            raise

    @staticmethod
    def _log_producer_done(sub_name: str, task: asyncio.Task) -> None:
        """Done-callback: surface a producer that died, the moment it dies.

        A producer only finishes with an exception once stamina has given up
        (or the error was never retryable), which means the subscription is
        gone for the rest of the night.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _logger.error(
                f"Event subscription '{sub_name}' died; no further events "
                f"will arrive from it.",
                exc_info=exc,
            )

    async def listen(self):
       """
       Starts and gathers all producer tasks, logging any producer that dies.
       """
       producer_tasks = []
       for source in self._sources:
           for sub_name, sub, client in source.subscriptions():
               task = asyncio.create_task(
                   self._producer(source.source_type, sub_name, sub, client)
               )
               task.add_done_callback(partial(self._log_producer_done, sub_name))
               producer_tasks.append(task)
       try:
           await asyncio.gather(*producer_tasks, return_exceptions=True)
       except asyncio.CancelledError:
           for task in producer_tasks:
               if not task.done():
                   task.cancel()
