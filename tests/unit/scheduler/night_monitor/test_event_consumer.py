# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from scheduler.night_monitor import EventConsumer
from scheduler.night_monitor.event_sources import EventSourceType


@pytest_asyncio.fixture
def queue():
    return asyncio.Queue()


@pytest_asyncio.fixture
def event_consumer(queue):
    """
    Provides an EventConsumer instance.
    Patches handlers so the real ones are not called.
    """
    with patch('scheduler.night_monitor.event_consumer.ResourceEventHandler', new_callable=MagicMock) as mock_resource_h, \
            patch('scheduler.night_monitor.event_consumer.WeatherEventHandler', new_callable=MagicMock) as mock_weather_h, \
            patch('scheduler.night_monitor.event_consumer.ODBEventHandler', new_callable=MagicMock) as mock_odb_h:
        # Configure the 'handle' method on the *mock instances* to be async
        mock_resource_h.return_value.handle = AsyncMock()
        mock_weather_h.return_value.handle = AsyncMock()
        mock_odb_h.return_value.handle = AsyncMock()

        c = EventConsumer(queue, shutdown_event=asyncio.Event())
        yield c

@pytest.mark.asyncio
async def test_match_source_unknown_raises_error(event_consumer):
    """Test that an unknown source raises a RuntimeError."""
    with pytest.raises(RuntimeError):
        event_consumer._match_source_to_handler("INVALID_SOURCE")


@pytest.mark.asyncio
async def test_consume_one_item(event_consumer, queue):
    """Test that the consumer correctly processes one item."""
    item = (EventSourceType.ODB, 'observation_edit', {"id": 1})
    mock_event = {"parsed_id": 1}

    # Configure the mock handler
    handler = event_consumer.odb_handler
    handler.parse_event.return_value = mock_event

    await queue.put(item)

    consumer_task = asyncio.create_task(event_consumer.consume())
    await queue.join()
    event_consumer._shutdown_event.set()
    await asyncio.sleep(0.01)

    # Check that the correct handler was used
    handler = event_consumer.odb_handler
    handler.handle.assert_called_once_with(item[1], item[2])

    # Check other handlers were not called
    event_consumer.weather_handler.handle.assert_not_called()

    # Kill just in case
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass




@pytest.mark.asyncio
async def test_consume_multiple_items(event_consumer, queue):
    """Test processing multiple items for different handlers."""
    item1 = (EventSourceType.RESOURCE, 'resource_edit', {"id": 1})
    item2 = (EventSourceType.WEATHER, 'weather_edit', {"id": 99})

    await queue.put(item1)
    await queue.put(item2)

    consumer_task = asyncio.create_task(event_consumer.consume())
    await queue.join()
    event_consumer._shutdown_event.set()
    await asyncio.sleep(0.01)

    # Check resource handler
    event_consumer.resource_handler.handle.assert_called_once_with(item1[1], item1[2])

    # Check weather handler
    event_consumer.weather_handler.handle.assert_called_once_with(item2[1], item2[2])

    # Check ODB handler
    event_consumer.odb_handler.handle.assert_not_called()

    # Kill just in case
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass