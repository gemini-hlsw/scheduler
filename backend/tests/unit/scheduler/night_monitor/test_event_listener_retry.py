# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""A transient ODB failure must reconnect, not end the night.

There is no replay: the obscalcUpdate subscription takes no cursor and gpp-client opens a
fresh subscription id on every connect, so a producer that dies stops delivering events
for the rest of the night. Anything transient therefore has to be retryable, and the
retryable set had holes that a loop stall would walk straight into.
"""

import asyncio
import socket
from unittest.mock import MagicMock, patch

import pytest
from gpp_client.generated.exceptions import (GraphQLClientError,
                                             GraphQLClientGraphQLMultiError,
                                             GraphQLClientInvalidMessageFormat)
from pydantic import ValidationError
from websockets import ConnectionClosedError, ConnectionClosedOK, InvalidStatus

from scheduler.night_monitor import EventListener, SubscriptionEndedException
from scheduler.night_monitor.event_listener import (RETRYABLE_EXCEPTIONS,
                                                    SourceMisconfiguredException)
from scheduler.night_monitor.event_sources import EventSourceType

_MODULE = 'scheduler.night_monitor.event_listener'


@pytest.mark.parametrize("exc,why", [
    # gpp-client turns the 5s graphql connection-ack timeout into this, destroying the
    # retryability the underlying asyncio.TimeoutError had. A stalled loop hits it.
    (GraphQLClientError, "the laundered 5s connection-ack timeout"),
    (GraphQLClientGraphQLMultiError, "a server error frame"),
    (GraphQLClientInvalidMessageFormat, "a malformed frame"),
    # gaierror is an OSError but NOT a ConnectionError, so it used to be fatal.
    (socket.gaierror, "a DNS blip on reconnect"),
    (ConnectionResetError, "the socket being reset"),
    (asyncio.TimeoutError, "the websocket open timeout"),
    (ConnectionClosedError, "an unclean close"),
    (ConnectionClosedOK, "a clean close of a long-lived subscription"),
    (InvalidStatus, "a non-101 handshake response"),
    (SubscriptionEndedException, "a server-side complete frame"),
    (ValidationError, "one payload that does not match the generated model"),
])
def test_transient_failures_are_retryable(exc, why):
    assert issubclass(exc, RETRYABLE_EXCEPTIONS), (
        f"{exc.__name__} ({why}) is not retryable, so it would end the subscription for "
        f"the rest of the night"
    )


def test_a_missing_client_is_not_retryable():
    # The one genuinely fatal case: retrying cannot conjure a client. It must stay
    # distinguishable from a bad payload, which is why it is no longer a bare ValueError.
    assert not issubclass(SourceMisconfiguredException, RETRYABLE_EXCEPTIONS)


def test_validation_error_is_not_confused_with_a_config_error():
    """The bug this replaced: pydantic's ValidationError IS a ValueError.

    The old `except ValueError: raise` existed to make a missing client fatal, and caught
    malformed payloads with it -- so one bad ODB event killed the night's subscription.
    """
    assert issubclass(ValidationError, ValueError)
    assert not issubclass(ValidationError, SourceMisconfiguredException)


@pytest.mark.asyncio
async def test_a_malformed_payload_reconnects_instead_of_killing_the_producer():
    listener = EventListener.__new__(EventListener)
    listener.queue = asyncio.Queue()
    listener._shutdown_event = asyncio.Event()

    def factory(_client):
        async def generator():
            raise ValidationError.from_exception_data("Obs", [])
            yield  # pragma: no cover - makes this an async generator
        return generator()

    # __wrapped__ reaches the undecorated body, so stamina does not retry here and the
    # exception type is observable.
    with pytest.raises(ValidationError):
        await EventListener._producer.__wrapped__(
            listener, EventSourceType.ODB, 'observation_edit', factory, None
        )


@pytest.mark.asyncio
async def test_a_missing_weather_client_is_reported_as_misconfiguration():
    listener = EventListener.__new__(EventListener)
    listener.queue = asyncio.Queue()
    listener._shutdown_event = asyncio.Event()

    with pytest.raises(SourceMisconfiguredException):
        await EventListener._producer.__wrapped__(
            listener, EventSourceType.WEATHER, 'weather_change', lambda c: None, None
        )


# --- what the log has to tell you ----------------------------------------------------
#
# A run of reconnects was unreadable because "Listening to X" was logged before the socket
# existed, so a failed connect and a connection that lived 26s produced identical lines.


def _listener() -> EventListener:
    listener = EventListener.__new__(EventListener)
    listener.queue = asyncio.Queue()
    listener._shutdown_event = asyncio.Event()
    return listener


def _generator_factory(*items, raising=None):
    def factory(_client):
        async def generator():
            for item in items:
                yield item
            if raising is not None:
                raise raising
        return generator()
    return factory


@pytest.mark.asyncio
async def test_a_drop_before_connecting_is_distinguishable_from_one_after():
    """The two failures in the real log: one never connected, one had been connected."""
    listener = _listener()

    with patch(f'{_MODULE}._logger') as logger:
        with pytest.raises(ConnectionClosedError):
            await EventListener._producer.__wrapped__(
                listener, EventSourceType.ODB, 'observation_edit',
                _generator_factory(raising=ConnectionClosedError(None, None)), None)
    never_connected = str(logger.warning.call_args)

    listener = _listener()
    with patch(f'{_MODULE}._logger') as logger:
        with pytest.raises(ConnectionClosedError):
            await EventListener._producer.__wrapped__(
                listener, EventSourceType.ODB, 'observation_edit',
                _generator_factory('event', raising=ConnectionClosedError(None, None)), None)
    was_connected = str(logger.warning.call_args)

    assert 'before delivering anything' in never_connected
    assert 'while connected' in was_connected
    assert 'dropped after' in never_connected and 'dropped after' in was_connected


@pytest.mark.asyncio
async def test_a_delivering_subscription_says_so_once():
    # The confirmation that "Opening subscription" never gave: proof it actually connected.
    listener = _listener()

    with patch(f'{_MODULE}._logger') as logger:
        with pytest.raises(SubscriptionEndedException):
            await EventListener._producer.__wrapped__(
                listener, EventSourceType.ODB, 'observation_edit',
                _generator_factory('a', 'b', 'c'), None)

    live = [c for c in logger.info.call_args_list if 'is live' in str(c)]
    assert len(live) == 1, f'expected exactly one live line for three events, got {len(live)}'


def test_the_odb_subscription_overrides_the_handshake_timeout():
    """websockets defaults open_timeout to 10s, which cost an extra reconnect.

    Observed: a connection died at 26s, the immediate retry failed at exactly 10.003s,
    and the next one succeeded -- so the ceiling, not the server, caused that failure.
    """
    from scheduler.night_monitor.event_sources import ODB_OPEN_TIMEOUT, ODBEventSource

    assert ODB_OPEN_TIMEOUT > 10.0, 'the whole point is to beat the 10s default'

    client = MagicMock()
    _name, factory, _c = ODBEventSource(client).subscriptions()[0]
    factory(None)

    kwargs = client._graphql.scheduler_observations_updates.call_args.kwargs
    assert kwargs['open_timeout'] == ODB_OPEN_TIMEOUT
    # Still the scheduler-scoped subscription, not the broader obscalc one.
    assert kwargs['executable_only'] is True
