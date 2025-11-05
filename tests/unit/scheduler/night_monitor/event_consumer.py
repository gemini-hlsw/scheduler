import asyncio

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from scheduler.night_monitor import EventConsumer, EventSourceType


@pytest_asyncio.fixture
def queue():
    """Provides a fresh asyncio.Queue for each test."""
    return asyncio.Queue()


@pytest_asyncio.fixture
def event_consumer(queue):
    """
    Provides an EventConsumer instance.

    This fixture patches the handler classes at their source, so when
    EventConsumer() is called, it gets our mocks instead of real handlers.
    """
    with patch('scheduler.night_monitor.event_consumer.ResourceEventHandler', new_callable=MagicMock) as mock_res_h, \
            patch('scheduler.night_monitor.event_consumer.WeatherEventHandler', new_callable=MagicMock) as mock_wea_h, \
            patch('scheduler.night_monitor.event_consumer.ODBEventHandler', new_callable=MagicMock) as mock_odb_h:
        # Configure the 'handle' method on the *mock instances* to be async
        mock_res_h.return_value.handle = AsyncMock()
        mock_wea_h.return_value.handle = AsyncMock()
        mock_odb_h.return_value.handle = AsyncMock()

        c = EventConsumer(queue)
        yield c

@pytest.mark.asyncio
async def test_match_source_unknown_raises_error(event_consumer):
    """Test that an unknown source raises a RuntimeError."""
    with pytest.raises(RuntimeError):
        await event_consumer._match_source_to_handler("INVALID_SOURCE")


@pytest.mark.asyncio
async def test_consume_one_item(event_consumer, queue):
    """Test that the consumer correctly processes one item."""
    item = (EventSourceType.RESOURCE, {"id": 1})
    mock_event = {"parsed_id": 1}

    # Configure the mock handler
    handler = event_consumer.resource_handler
    handler.parse_event.return_value = mock_event

    await queue.put(item)
    await queue.put(None)

    await event_consumer.consume()

    # Check that the correct handler was used
    handler.parse_event.assert_called_once_with(item[1])
    handler.handle.assert_called_once_with(mock_event)

    # Check other handlers were not called
    event_consumer.weather_handler.parse_event.assert_not_called()


@pytest.mark.asyncio
async def test_consume_multiple_items(event_consumer, queue):
    """Test processing multiple items for different handlers."""
    item1 = (EventSourceType.RESOURCE, {"id": 1})
    item2 = (EventSourceType.WEATHER, {"id": 99})

    await queue.put(item1)
    await queue.put(item2)
    await queue.put(None)

    await event_consumer.consume()

    # Check resource handler
    event_consumer.resource_handler.parse_event.assert_called_once_with(item1[1])
    event_consumer.resource_handler.handle.assert_called_once()

    # Check weather handler
    event_consumer.weather_handler.parse_event.assert_called_once_with(item2[1])
    event_consumer.weather_handler.handle.assert_called_once()

    # Check ODB handler
    event_consumer.odb_handler.parse_event.assert_not_called()
