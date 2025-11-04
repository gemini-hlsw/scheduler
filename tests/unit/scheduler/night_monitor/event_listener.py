# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from unittest.mock import AsyncMock, patch, call

import pytest
import asyncio

from scheduler.night_monitor import EventListener, EventSourceType, SubscriptionEndedException

async def mock_subscription_generator(data_list):
    """A mock async generator that yields items from data_list."""
    for item in data_list:
        yield item


async def mock_failing_subscription_generator(data_list, error_to_raise, fail_count=1):
    failed_attempts = 0
    while failed_attempts < fail_count:
        failed_attempts += 1
        raise error_to_raise("Simulated connection error")  # <--- PROBLEM

    # This code is unreachable when fail_count > 0
    for item in data_list:
        yield item

@pytest.fixture
def mock_client_async():
    """Provides an AsyncMock for asynchronous tests."""
    return AsyncMock()

@pytest.fixture
def event_listener(mock_client_async):
    """Provides an EventListener instance initialized with the async mock client."""
    return EventListener(mock_client_async)

@pytest.mark.asyncio
async def test_produce_success(event_listener):
    """Test the _producer method successfully reeds from subscription"""

    # Mock subscription factory
    mock_data = ['data1', 'data2']
    mock_sub_factory = AsyncMock(return_value=mock_subscription_generator(mock_data))
    source_type = EventSourceType.RESOURCE

    await event_listener._producer(source_type, mock_sub_factory)

    # Check if the factory was called
    mock_sub_factory.assert_called_once()

    # Check the queue
    assert event_listener.queue.qsize() == 2
    assert await event_listener.queue.get() == (source_type, 'data1')
    assert await event_listener.queue.get() == (source_type, 'data2')


@patch('stamina.retry', lambda **kwargs: lambda f: f) # Disable stamina
@pytest.mark.asyncio
async def test_producer_ends_gracefully(event_listener):
    """Test that the producer correctly raises SubscriptionEndedException."""
    mock_data = ['data1']
    mock_sub_factory = AsyncMock(return_value=mock_subscription_generator(mock_data))
    source_type = EventSourceType.RESOURCE

    # The producer should raise SubscriptionEndedException when the generator finishes
    await event_listener._producer(source_type, mock_sub_factory)

    # Check that the data was still processed
    assert event_listener.queue.qsize() == 1
    assert await event_listener.queue.get() == (source_type, 'data1')

@patch('asyncio.sleep', new_callable=AsyncMock)  # Patch sleep to speed up retry
@pytest.mark.asyncio
async def test_producer_retry(mock_sleep, event_listener):
    """Test the producer's retry logic via stamina."""
    mock_data = ['data1']
    error_to_raise = ConnectionError

    # This generator will fail once, then succeed
    mock_sub_factory = AsyncMock(side_effect=[
        error_to_raise("Simulated connection error"),  # 1st call: fails
        mock_subscription_generator(mock_data)  # 2nd call: succeeds
    ])
    source_type = EventSourceType.WEATHER

    # It should eventually succeed and then raise SubscriptionEndedException
    await event_listener._producer(source_type, mock_sub_factory)

    # 1 fail, 1 success
    assert mock_sub_factory.call_count == 2

    # Check that the data was processed after the retry
    assert event_listener.queue.qsize() == 1
    assert await event_listener.queue.get() == (source_type, 'data1')

    # Check if was called for the retry
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_listen(event_listener, mock_client_async):
    """Test the main listen() method to ensure all producers are gathered."""
    # Mock the subscription generators
    mock_res_gen = mock_subscription_generator(['res_data'])
    mock_weather_gen = mock_subscription_generator(['weather_data'])
    mock_odb1_gen = mock_subscription_generator(['odb_data1'])
    mock_odb2_gen = mock_subscription_generator(['odb_data2'])

    mock_client_async.subscribe.side_effect = [
        mock_res_gen,
        mock_weather_gen,
        mock_odb1_gen,
        mock_odb2_gen
    ]

    listen_task = asyncio.create_task(event_listener.listen())
    await listen_task

    # Check that all subscriptions were created
    mock_client_async.subscribe.assert_has_calls([
        call('resource_edit'),
        call('weather_change'),
        call('observation_edit'),
        call('observation_change')
    ])

    # Check the queue for all items
    assert event_listener.queue.qsize() == 4

    # We get all items and check the set of items
    items_in_queue = set()
    while not event_listener.queue.empty():
        items_in_queue.add(await event_listener.queue.get())

    expected_items = {
        (EventSourceType.RESOURCE, 'res_data'),
        (EventSourceType.WEATHER, 'weather_data'),
        (EventSourceType.ODB, 'odb_data1'),
        (EventSourceType.ODB, 'odb_data2')
    }
    assert items_in_queue == expected_items
