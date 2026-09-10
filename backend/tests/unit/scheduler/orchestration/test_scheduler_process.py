# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scheduler.orchestration.scheduler_process import SchedulerProcess


async def _sleep_forever():
    await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_stop_process_cleans_up_everything():
    """stop_process must cancel the run task AND the engine task, and shut
    down the night monitor — not just self.task."""
    process = SchedulerProcess("test", MagicMock())

    run_task = asyncio.create_task(_sleep_forever())
    engine_task = asyncio.create_task(_sleep_forever())
    process.task = run_task
    process._engine_task = engine_task
    process.night_monitor = MagicMock()
    process.night_monitor.shutdown = AsyncMock()
    process.running_event.set()

    await process.stop_process()

    assert run_task.done(), "run task still alive after stop_process"
    assert engine_task.done(), "engine task leaked by stop_process"
    process.night_monitor.shutdown.assert_awaited_once()
    assert not process.running_event.is_set()


@pytest.mark.asyncio
async def test_stop_process_before_start_does_not_crash():
    """Stopping a process that never started is a no-op, not a crash."""
    process = SchedulerProcess("test", MagicMock())
    await process.stop_process()
    assert not process.running_event.is_set()


@pytest.mark.asyncio
async def test_night_monitor_shutdown_cancels_night_tracker():
    """NightMonitor.shutdown must also stop the night tracker task, which has
    no shutdown_event of its own."""
    from scheduler.night_monitor.night_monitor import NightMonitor

    with patch('scheduler.night_monitor.night_monitor.gpp'), \
         patch('scheduler.night_monitor.night_monitor.EventListener'), \
         patch('scheduler.night_monitor.night_monitor.EventConsumer'), \
         patch('scheduler.night_monitor.night_monitor.NightTracker'):
        monitor = NightMonitor(MagicMock(), frozenset(), MagicMock(), MagicMock())

    monitor._listener_task = asyncio.create_task(_sleep_forever())
    monitor._consumer_task = asyncio.create_task(_sleep_forever())
    monitor._night_tracker_task = asyncio.create_task(_sleep_forever())

    await monitor.shutdown(drain_queue=False)

    assert monitor._night_tracker_task.done(), "night tracker task leaked by shutdown"
    assert monitor._listener_task.done()
    assert monitor._consumer_task.done()


async def _start_process_with_engine(process, engine_run_coro):
    """Run SchedulerProcess.run() with NightMonitor/EngineRT mocked out, the
    engine task backed by engine_run_coro."""
    with patch('scheduler.orchestration.scheduler_process.NightMonitor') as mock_nm, \
         patch('scheduler.orchestration.scheduler_process.EngineRT') as mock_engine:
        mock_nm.return_value.start = AsyncMock()
        mock_nm.return_value.shutdown = AsyncMock()
        mock_engine.return_value.run = engine_run_coro
        await process.run()


@pytest.mark.asyncio
async def test_engine_crash_is_logged_published_and_stops_running():
    """A crash in the engine task must be logged, push a NightPlansError to
    the process's subscribers, and clear running_event."""
    from scheduler.graphql_mid.types import NightPlansError
    from scheduler.shared_queue import plan_response_subscribers

    process = SchedulerProcess("crash-test", MagicMock())
    subscriber_queue = asyncio.Queue()
    plan_response_subscribers["crash-test"] = {subscriber_queue}

    async def crashing_engine():
        raise RuntimeError("engine exploded")

    try:
        with patch('scheduler.orchestration.scheduler_process._logger') as mock_logger:
            await _start_process_with_engine(process, crashing_engine)
            assert process.running_event.is_set()

            # Let the engine task die and the supervision callback fire.
            await asyncio.sleep(0.01)

        assert not process.running_event.is_set(), "running_event still set after engine crash"
        mock_logger.error.assert_called()

        published = subscriber_queue.get_nowait()
        assert isinstance(published, NightPlansError)
        assert "engine exploded" in published.error
    finally:
        plan_response_subscribers.pop("crash-test", None)


@pytest.mark.asyncio
async def test_engine_cancellation_is_not_reported_as_crash():
    """Cancelling the engine (normal shutdown) must not publish errors."""
    from scheduler.shared_queue import plan_response_subscribers

    process = SchedulerProcess("cancel-test", MagicMock())
    subscriber_queue = asyncio.Queue()
    plan_response_subscribers["cancel-test"] = {subscriber_queue}

    async def long_engine():
        await asyncio.sleep(3600)

    try:
        with patch('scheduler.orchestration.scheduler_process._logger') as mock_logger:
            await _start_process_with_engine(process, long_engine)
            process._engine_task.cancel()
            await asyncio.sleep(0.01)

        mock_logger.error.assert_not_called()
        assert subscriber_queue.empty()
    finally:
        plan_response_subscribers.pop("cancel-test", None)
