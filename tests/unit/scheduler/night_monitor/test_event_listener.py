# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import asyncio

from scheduler.night_monitor import EventListener, SubscriptionEndedException
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

@patch('scheduler.night_monitor.event_listener.stamina.retry', lambda **kwargs: lambda f: f) # Disable stamina
@pytest.mark.asyncio
async def test_produce_success(event_listener):
    """Test the _producer method successfully reeds from subscription"""

    # Mock subscription factory
    mock_data = ['data1', 'data2']
    mock_sub_factory = AsyncMock(return_value=mock_subscription_generator(mock_data))
    source_type = EventSourceType.RESOURCE

    with pytest.raises(SubscriptionEndedException):
        await event_listener._producer(source_type, 'resource_edit', mock_sub_factory, )

    # Check if the factory was called
    mock_sub_factory.assert_called_once()

    # Check the queue
    assert event_listener.queue.qsize() == 2, f"queue size should be 2 events but it got {event_listener.queue.qsize()}"
    assert await event_listener.queue.get() == (source_type, 'resource_edit', 'data1')
    assert await event_listener.queue.get() == (source_type, 'resource_edit', 'data2')


@patch('scheduler.night_monitor.event_listener.stamina.retry', lambda **kwargs: lambda f: f) # Disable stamina
@pytest.mark.asyncio
async def test_producer_ends_gracefully(event_listener):
    """Test that the producer correctly raises SubscriptionEndedException."""
    mock_data = ['data1']
    mock_sub_factory = AsyncMock(return_value=mock_subscription_generator(mock_data))
    source_type = EventSourceType.RESOURCE

    # The producer should raise SubscriptionEndedException when the generator finishes
    with pytest.raises(SubscriptionEndedException):
        await event_listener._producer(source_type, 'resource_edit', mock_sub_factory)

    # Check that the data was still processed
    assert event_listener.queue.qsize() == 1, f"queue size should be 1 event but it got {event_listener.queue.qsize()}"
    assert await event_listener.queue.get() == (source_type, 'resource_edit', 'data1'), "message retrieved is not the same as the one queued"

@patch('asyncio.sleep', new_callable=AsyncMock)  # Patch sleep to speed up retry
@pytest.mark.asyncio
async def test_producer_retry(mock_sleep, event_listener):
    """Test the producer's retry logic via stamina."""
    mock_data = ['data1']
    # Mock factory that fails once with a retryable exception, then succeeds
    mock_sub_factory = AsyncMock()

    mock_sub_factory.side_effect = [
        ConnectionError("Simulated connection error"),
        mock_subscription_generator(mock_data)
    ]

    source_type = EventSourceType.WEATHER

    # It should eventually succeed and then raise SubscriptionEndedException
    with pytest.raises(SubscriptionEndedException):
        await event_listener._producer(source_type, 'weather_edit', mock_sub_factory)

    # 1 fail, 1 success
    assert mock_sub_factory.call_count == 2

    # Check that the data was processed after the retry
    assert event_listener.queue.qsize() == 1, f"queue size should be 1 event but it got {event_listener.queue.qsize()}"
    assert await event_listener.queue.get() == (source_type, 'weather_edit', 'data1')

    # Check if was called for the retry
    mock_sleep.assert_called_once()


@patch('scheduler.night_monitor.event_listener.stamina.retry', lambda **kwargs: lambda f: f)
@pytest.mark.asyncio
async def test_listen(event_listener, mock_client_async):
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
    async def resource_factory():
        return mock_subscription_generator(['res_data'])

    async def weather_factory():
        return mock_subscription_generator(['weather_data'])

    async def odb_factory1():
        return mock_subscription_generator(['odb_data1'])

    async def odb_factory2():
        return mock_subscription_generator(['odb_data2'])

    mock_resource_source.subscriptions.return_value = [
        ('resource_edit', resource_factory)
    ]

    mock_weather_source.subscriptions.return_value = [
        ('weather_change', weather_factory)
    ]

    mock_odb_source.subscriptions.return_value = [
        ('observation_edit', odb_factory1),
        ('observation_edit', odb_factory2)
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