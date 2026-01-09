import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from scheduler.clients import SchedulerEvent, SchedulerQueue
from scheduler.night_monitor.event_handlers.resource_event_handler import MockResourceEvent, PDRQueue, ResourceEventHandler


class TestPDRQueue:
    """Tests for PDRQueue class."""

    @pytest.fixture
    def queue(self):
        return PDRQueue(timeout=15.0)

    @pytest.fixture
    def sample_event(self):
        return MockResourceEvent(resource_status="disabled", resource_name="test_resource")

    @pytest.fixture
    def schedule_queue(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_put_adds_event_to_queue(self, queue, sample_event, schedule_queue):
        callback = AsyncMock()
        await queue.put(sample_event, schedule_queue, callback)

        assert "test_resource" in queue.queue
        assert queue.queue["test_resource"] == sample_event
        assert "test_resource" in queue.tasks

    @pytest.mark.asyncio
    async def test_put_creates_auto_pop_task(self, queue, sample_event, schedule_queue):
        callback = AsyncMock()
        await queue.put(sample_event, schedule_queue, callback)

        assert "test_resource" in queue.tasks
        assert isinstance(queue.tasks["test_resource"], asyncio.Task)

    @pytest.mark.asyncio
    async def test_auto_pop_triggers_after_timeout(self, sample_event, schedule_queue):
        pdr = PDRQueue(timeout=0.1)
        callback = AsyncMock()

        await pdr.put(sample_event, schedule_queue, callback, timeout=0.1)
        await asyncio.sleep(0.2)

        assert "test_resource" not in pdr.queue
        assert "test_resource" not in pdr.tasks
        callback.assert_called_once_with(sample_event, schedule_queue)

    @pytest.mark.asyncio
    async def test_force_pop_removes_event(self, queue, sample_event, schedule_queue):
        callback = AsyncMock()
        await queue.put(sample_event, schedule_queue, callback)

        await queue.force_pop("test_resource")

        assert "test_resource" not in queue.queue
        assert "test_resource" not in queue.tasks

    @pytest.mark.asyncio
    async def test_force_pop_cancels_auto_pop_task(self, queue, sample_event, schedule_queue):
        callback = AsyncMock()
        await queue.put(sample_event, schedule_queue, callback, timeout=0.5)

        await queue.force_pop("test_resource")
        await asyncio.sleep(0.6)

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_removes_all_items(self, queue, schedule_queue):
        callback = AsyncMock()
        event1 = MockResourceEvent(resource_status="disabled", resource_name="test_resource")
        event2 = MockResourceEvent(resource_status="disabled", resource_name="resource_2")

        await queue.put(event1, schedule_queue, callback)
        await queue.put(event2, schedule_queue, callback)

        await queue.clear()

        assert len(queue.queue) == 0
        assert len(queue.tasks) == 0

class TestResourceEventHandler:
    """Tests for ResourceEventHandler class."""

    @pytest.fixture
    def handler(self):
        return ResourceEventHandler(scheduler_queue=SchedulerQueue())

    @pytest.fixture
    def schedule_queue(self):
        return AsyncMock()

    @pytest.fixture
    def mock_last_plan(self):
        """Patch LastPlanMock to return specified resources."""

        def _mock(resources: list[str]):
            mock = MagicMock()
            mock.resources.return_value = resources
            return patch('scheduler.night_monitor.event_handlers.resource_event_handler.LastPlanMock', return_value=mock)

        return _mock

    @pytest.mark.asyncio
    async def test_on_resource_edit_enabled_removes_from_pdr(self, handler, mock_last_plan, schedule_queue):
        disabled_event = MockResourceEvent(resource_status="disabled", resource_name="test_resource")
        enabled_event = MockResourceEvent(resource_status="enabled", resource_name="test_resource")
        with mock_last_plan(["test_resource"]):
            await handler._on_resource_edit(disabled_event)
            await asyncio.sleep(0)
            assert "test_resource" in handler.pdr.queue

            await handler._on_resource_edit(enabled_event)

        assert "test_resource" not in handler.pdr.queue

    @pytest.mark.asyncio
    async def test_on_resource_edit_unknown_status_raises(self, handler, schedule_queue):
        event = MockResourceEvent(resource_status="unknown", resource_name="test_resource")

        with pytest.raises(RuntimeError, match="Unknown resource status"):
            await handler._on_resource_edit(event)

    @pytest.mark.asyncio
    async def test_enabled_before_timeout_prevents_callback(self, handler, schedule_queue):
        handler.pdr = PDRQueue(timeout=0.2)

        with patch.object(handler, '_disabled_callback', new_callable=AsyncMock) as mock_cb:
            handler.pdr = PDRQueue(timeout=0.2)
            disabled_event = MockResourceEvent(resource_status="disabled", resource_name="test_resource")
            enabled_event = MockResourceEvent(resource_status="enabled", resource_name="test_resource")

            await handler._on_resource_edit(disabled_event)
            await asyncio.sleep(0.05)
            await handler._on_resource_edit(enabled_event)
            await asyncio.sleep(0.3)

            # Callback should not have been called since we enabled before timeout
            assert "test_resource" not in handler.pdr.queue

    @pytest.mark.asyncio
    async def test_timeout_triggers_callback(self, handler, schedule_queue):
        handler.pdr = PDRQueue(timeout=0.1)
        callback_called = asyncio.Event()
        received_event = None

        async def test_callback(event: MockResourceEvent, sched_queue: SchedulerQueue):
            nonlocal received_event
            received_event = event
            callback_called.set()

        disabled = MockResourceEvent(resource_status="disabled", resource_name="test_resource")
        await handler.pdr.put(disabled, schedule_queue, test_callback, timeout=0.1)

        await asyncio.wait_for(callback_called.wait(), timeout=0.5)

        assert received_event == disabled
        assert "test_resource" not in handler.pdr.queue
