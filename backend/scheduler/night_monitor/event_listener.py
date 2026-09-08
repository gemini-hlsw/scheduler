# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from functools import partial
from time import perf_counter

import stamina
from aiohttp import ClientError
from gpp_client.generated.exceptions import GraphQLClientError
from pydantic import ValidationError
from websockets import ConnectionClosedError, ConnectionClosedOK, InvalidStatus, WebSocketException
from typing import Any

from .event_sources import (
    ODB_OPEN_TIMEOUT,
    ResourceEventSource,
    WeatherEventSource,
    ODBEventSource,
    EventSourceType,
)

from scheduler.services import logger_factory
_logger = logger_factory.create_logger(__name__)


__all__ = ['EventListener', 'SubscriptionEndedException', 'SourceMisconfiguredException']


class SubscriptionEndedException(Exception): pass
# SubscriptionEndedException is raised below when a subscription ends without an
# error, which is what a server-side `complete` frame looks like.


class SourceMisconfiguredException(Exception): pass
# A source was wired up without the client it needs.


RETRYABLE_EXCEPTIONS = (
    OSError, asyncio.TimeoutError,
    ClientError, ConnectionClosedError,
    ConnectionClosedOK, InvalidStatus,
    WebSocketException,
    GraphQLClientError,
    ValidationError,
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
        # How long this attempt lasted is the single most useful thing in the log. A drop
        # at ~10s is the websocket open_timeout, i.e. we never connected; a drop at 26s
        # means we were connected and the transport died under us. Without this the two
        # are indistinguishable, which is exactly what made a run of reconnects
        # unreadable.
        started = perf_counter()
        live = False

        def _mark_live() -> None:
            """Say once, on the first event, that the subscription is really delivering."""
            nonlocal live
            if not live:
                live = True
                _logger.info(
                    f"Subscription '{sub_name}' is live "
                    f"(first event after {perf_counter() - started:.1f}s)."
                )

        try:
            # Create the actual session
            if source == EventSourceType.WEATHER:
                if client is None:
                    raise SourceMisconfiguredException("Client is not initialized for WeatherEventSource.")
                async with client as session:
                    sub_generator = subscription_factory(session)
                    async for data in sub_generator:
                        _mark_live()
                        _logger.debug("Received Weather event:")
                        _logger.debug(data)
                        if self._shutdown_event.is_set():
                            break
                        await self.queue.put((source, sub_name, data))

                if not self._shutdown_event.is_set():
                    raise SubscriptionEndedException(f"Subscription '{sub_name}' ended gracefully, retrying.")

            else:
                # "Opening", not "Listening": this runs before the socket exists, so it is
                # an intent, not a confirmation. _mark_live() below is the confirmation.
                _logger.info(f"Opening subscription '{sub_name}'...")
                async for data in subscription_factory(client):
                    _mark_live()
                    if self._shutdown_event.is_set():
                        break
                    await self.queue.put((source, sub_name, data))

                if not self._shutdown_event.is_set():
                    raise SubscriptionEndedException(f"Subscription '{sub_name}' ended gracefully, retrying.")

        except SourceMisconfiguredException:
            raise

        except asyncio.CancelledError:
            raise

        except ValidationError as e:
            _logger.error(
                f"Subscription '{sub_name}' received a payload that does not match the "
                f"generated model after {perf_counter() - started:.1f}s; reconnecting past "
                f"it. Errors: {e.errors()}"
            )
            raise

        except RETRYABLE_EXCEPTIONS as e:
            # Retries are now unbounded, so a subscription that can never connect
            # would reconnect silently forever.
            elapsed = perf_counter() - started
            # Distinguishing these two is the whole point of the timing above.
            phase = ('while connected' if live
                     else f'before delivering anything (open_timeout is {ODB_OPEN_TIMEOUT:.0f}s)')
            _logger.warning(
                f"Subscription '{sub_name}' dropped after {elapsed:.1f}s {phase} "
                f"({type(e).__name__}: {e}); reconnecting."
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
