# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import asyncio

from websockets import ConnectionClosedError, ConnectionClosedOK

from scheduler.night_monitor import EventListener, SubscriptionEndedException
from scheduler.night_monitor.event_listener import RETRYABLE_EXCEPTIONS
from scheduler.night_monitor.event_sources import EventSourceType

async def mock_subscription_generator(data_list):
    """A mock async generator that yields items from data_list."""
    for item in data_list:
        yield item


@pytest.fixture(autouse=True)
def mock_client_async():
    """Provides an AsyncMock for asynchronous tests."""
    mock_client = AsyncMock()
    yield mock_client
    mock_client.reset_mock()

@pytest.fixture
def event_listener(mock_client_async):
    """Provides an EventListener instance initialized with the async mock client."""
    return EventListener(mock_client_async, asyncio.Queue(), asyncio.Event())


@pytest.fixture
def no_retry():
    """
    Run producers without the retry wrapper, for tests about what a producer does rather
    than about reconnecting.

    Patching ``stamina.retry`` does not work: the decorator is applied at import time, so
    replacing the name afterwards leaves the already-wrapped function in place. Retries are
    unbounded by design, so a producer whose subscription ends never returns unless it is
    unwrapped here.
    """
    with patch.object(EventListener, '_producer', EventListener._producer.__wrapped__):
        yield


@pytest.mark.asyncio
async def test_produce_success(no_retry, event_listener, mock_client_async):
    """Test the _producer method successfully reeds from subscription"""

    # Mock subscription factory - must return generator directly, not a coroutine
    # Use MagicMock (not AsyncMock) so we can still assert it was called
    mock_data = ['data1', 'data2']
    mock_sub_factory = MagicMock(side_effect=lambda session: mock_subscription_generator(mock_data))
    source_type = EventSourceType.RESOURCE

    client = AsyncMock()
    with pytest.raises(SubscriptionEndedException):
        await event_listener._producer(source_type, 'resource_edit', mock_sub_factory, client)

    # Check if the factory was called
    mock_sub_factory.assert_called_once()

    # Check the queue
    assert event_listener.queue.qsize() == 2, f"queue size should be 2 events but it got {event_listener.queue.qsize()}"
    assert await event_listener.queue.get() == (source_type, 'resource_edit', 'data1')
    assert await event_listener.queue.get() == (source_type, 'resource_edit', 'data2')


@pytest.mark.asyncio
async def test_producer_ends_gracefully(no_retry, event_listener,  mock_client_async):
    """Test that the producer correctly raises SubscriptionEndedException."""
    mock_data = ['data1']
    mock_sub_factory = MagicMock(side_effect=lambda session: mock_subscription_generator(mock_data))
    source_type = EventSourceType.RESOURCE

    client = AsyncMock()
    # The producer should raise SubscriptionEndedException when the generator finishes
    with pytest.raises(SubscriptionEndedException):
        await event_listener._producer(source_type, 'resource_edit', mock_sub_factory,  client)

    # Check that the data was still processed
    assert event_listener.queue.qsize() == 1, f"queue size should be 1 event but it got {event_listener.queue.qsize()}"
    assert await event_listener.queue.get() == (source_type, 'resource_edit', 'data1'), "message retrieved is not the same as the one queued"

@patch('asyncio.sleep', new_callable=AsyncMock)  # Patch sleep to speed up retry
@pytest.mark.asyncio
async def test_producer_retry(mock_sleep, event_listener):
    """Test the producer's retry logic via stamina."""
    mock_sub_factory = MagicMock()

    # Shutting down is the only clean way out of a live producer: a subscription that
    # merely ends is itself retryable, so without this it would reconnect forever.
    async def last_generator(session):
        yield 'data1'
        event_listener._shutdown_event.set()

    # Mock factory that fails once with a retryable exception, then succeeds
    mock_sub_factory.side_effect = [
        ConnectionError("Simulated connection error"),
        last_generator(None)
    ]

    source_type = EventSourceType.WEATHER

    client = AsyncMock()
    await event_listener._producer(source_type, 'weather_edit', mock_sub_factory, client)

    # 1 fail, 1 success
    assert mock_sub_factory.call_count == 2

    # Check that the data was processed after the retry
    assert event_listener.queue.qsize() == 1, f"queue size should be 1 event but it got {event_listener.queue.qsize()}"
    assert await event_listener.queue.get() == (source_type, 'weather_edit', 'data1')

    # Check if was called for the retry
    mock_sleep.assert_called_once()


@patch('asyncio.sleep', new_callable=AsyncMock)  # Patch sleep to speed up retry
@pytest.mark.asyncio
async def test_producer_reconnects_after_a_graceful_end(mock_sleep, event_listener):
    """
    A subscription that ends without an error must reconnect, not end the night.

    This is what a server-side `complete` frame looks like from here: the client closes the
    socket and the async-for simply runs out. Before, that raised a SubscriptionEndedException
    that nothing retried, and the ODB events stopped arriving until the process restarted.
    """
    async def ends_immediately(session):
        return
        yield  # pragma: no cover - makes this an async generator

    async def then_delivers(session):
        yield 'data1'
        event_listener._shutdown_event.set()

    mock_sub_factory = MagicMock()
    mock_sub_factory.side_effect = [ends_immediately(None), then_delivers(None)]

    await event_listener._producer(
        EventSourceType.ODB, 'observation_edit', mock_sub_factory, AsyncMock()
    )

    assert mock_sub_factory.call_count == 2, "a graceful end must reconnect"
    assert await event_listener.queue.get() == (EventSourceType.ODB, 'observation_edit', 'data1')


@pytest.mark.asyncio
async def test_graceful_end_and_clean_close_are_retryable():
    """
    Pins the two closures that end a healthy subscription.

    The elapsed-time half of this fix (stamina's default `timeout=45.0`, measured from when
    the call first started) cannot be asserted here without a >45s test; it is covered by the
    decorator's explicit `timeout=None`.
    """
    assert SubscriptionEndedException in RETRYABLE_EXCEPTIONS
    assert ConnectionClosedOK in RETRYABLE_EXCEPTIONS
    assert ConnectionClosedError in RETRYABLE_EXCEPTIONS


@pytest.mark.asyncio
async def test_listen(no_retry, event_listener, mock_client_async):
    """Test the main listen() method to ensure all producers are gathered."""

    mock_resource_source = MagicMock()
    mock_weather_source = MagicMock()
    mock_odb_source = MagicMock()

    # Set source types
    mock_resource_source.source_type = EventSourceType.RESOURCE
    mock_weather_source.source_type = EventSourceType.WEATHER
    mock_odb_source.source_type = EventSourceType.ODB

    # Mock subscriptions() to return (subscription_name, factory) tuples
    # The factory is a callable that returns an async generator when awaited
    # Factories must be lambdas that return generators directly, not async functions
    resource_factory = lambda session: mock_subscription_generator(['res_data'])
    weather_factory = lambda session: mock_subscription_generator(['weather_data'])
    odb_factory1 = lambda session: mock_subscription_generator(['odb_data1'])
    odb_factory2 = lambda session: mock_subscription_generator(['odb_data2'])

    # Subscriptions must return 3-tuples: (sub_name, factory, client)
    mock_resource_source.subscriptions.return_value = [
        ('resource_edit', resource_factory, mock_client_async)
    ]

    mock_weather_source.subscriptions.return_value = [
        ('weather_change', weather_factory, mock_client_async)
    ]

    mock_odb_source.subscriptions.return_value = [
        ('observation_edit', odb_factory1, mock_client_async),
        ('observation_edit', odb_factory2, mock_client_async)
    ]

    event_listener._sources = [mock_resource_source, mock_weather_source, mock_odb_source]

    await event_listener.listen()

    # Verify that subscriptions() was called on each source
    mock_resource_source.subscriptions.assert_called_once()
    mock_weather_source.subscriptions.assert_called_once()
    mock_odb_source.subscriptions.assert_called_once()

    # Check the queue contains all items
    assert event_listener.queue.qsize() == 4, \
        f"Expected 4 items in queue but got {event_listener.queue.qsize()}"

    # Get all items and verify them
    items_in_queue = set()
    while not event_listener.queue.empty():
        items_in_queue.add(await event_listener.queue.get())

    expected_items = {
        (EventSourceType.RESOURCE, 'resource_edit', 'res_data'),
        (EventSourceType.WEATHER, 'weather_change', 'weather_data'),
        (EventSourceType.ODB, 'observation_edit', 'odb_data1'),
        (EventSourceType.ODB, 'observation_edit', 'odb_data2')
    }
    assert items_in_queue == expected_items, \
        f"Queue items don't match. Got: {items_in_queue}, Expected: {expected_items}"

@pytest.mark.asyncio
async def test_listen_logs_producer_failure(no_retry, event_listener, mock_client_async):
    """A producer that dies must be logged, not silently swallowed by the gather.

    Unwrapped, because with retries in place a producer only reaches the done-callback once
    stamina has given up, which for a retryable error is never.
    """
    mock_source = MagicMock()
    mock_source.source_type = EventSourceType.ODB

    def failing_factory(session):
        raise RuntimeError("subscription exploded")

    ok_factory = lambda session: mock_subscription_generator(['odb_data'])

    mock_source.subscriptions.return_value = [
        ('observation_edit', failing_factory, mock_client_async),
        ('program_edit', ok_factory, mock_client_async),
    ]
    event_listener._sources = [mock_source]

    with patch('scheduler.night_monitor.event_listener._logger') as mock_logger:
        await event_listener.listen()

    error_messages = [str(call) for call in mock_logger.error.call_args_list]
    assert any('observation_edit' in message for message in error_messages), \
        f"Expected an error log for the dead 'observation_edit' producer, got: {error_messages}"


@pytest.mark.asyncio
async def test_listen_cancellation_is_not_logged_as_failure(event_listener, mock_client_async):
    """Shutdown cancellation is not a producer death: no error log."""
    mock_source = MagicMock()
    mock_source.source_type = EventSourceType.ODB

    async def never_ending_generator(session):
        while True:
            await asyncio.sleep(3600)
            yield 'never'

    mock_source.subscriptions.return_value = [
        ('observation_edit', never_ending_generator, mock_client_async),
    ]
    event_listener._sources = [mock_source]

    with patch('scheduler.night_monitor.event_listener._logger') as mock_logger:
        listen_task = asyncio.create_task(event_listener.listen())
        await asyncio.sleep(0.01)
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass

    mock_logger.error.assert_not_called()
