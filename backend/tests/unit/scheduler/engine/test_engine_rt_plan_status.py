# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Every UI is told when the Engine computes a plan, not only the one that asked for it."""

import asyncio
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import Site

from scheduler.core.events.queue.events import OnDemandScheduleEvent
from scheduler.core.events.queue.plan_status import PlanState, PlanStatusTracker
from scheduler.engine.engineRT import EngineRT

_MODULE = "scheduler.engine.engineRT"


@pytest.fixture(autouse=True)
def _coordination():
    with patch(f"{_MODULE}.coordination") as coord:
        coord.wait_until_aggregator_idle = AsyncMock()
        coord.signal_plan_in_progress = AsyncMock()
        coord.signal_plan_done = AsyncMock()
        yield coord


def _engine() -> EngineRT:
    params = MagicMock()
    params.sites = frozenset({Site.GN})
    return EngineRT(
        params,
        scheduler_queue=MagicMock(),
        process_id="test",
        weather_source=MagicMock(),
        nightly_timeline_store=MagicMock(),
        plan_status=PlanStatusTracker(),
    )


def _event(requested_by_user: bool = False) -> OnDemandScheduleEvent:
    return OnDemandScheduleEvent(
        site=None, time=datetime.now(UTC), description="Plan requested", requested_by_user=requested_by_user
    )


def _states(queue: asyncio.Queue) -> list[PlanState]:
    states = []
    while not queue.empty():
        states.append(queue.get_nowait().state)
    return states


@pytest.mark.asyncio
async def test_a_plan_goes_waiting_computing_idle(_coordination):
    engine = _engine()
    seen_while_computing = []

    async def compute(_):
        seen_while_computing.append(engine.plan_status.current)
        return "plans"

    engine._compute_event_plan = compute
    with engine.plan_status.subscribe() as queue:
        assert await engine.compute_event_plan(_event(requested_by_user=True)) == "plans"

    assert _states(queue) == [PlanState.WAITING, PlanState.COMPUTING, PlanState.IDLE]
    (computing,) = seen_while_computing
    assert computing.state is PlanState.COMPUTING
    assert computing.event_description == "Plan requested"
    assert computing.requested_by_user

    done = engine.plan_status.current
    assert done.state is PlanState.IDLE
    assert done.last_error is None
    assert done.last_duration_seconds is not None and done.last_duration_seconds >= 0
    assert done.event_description is None


@pytest.mark.asyncio
async def test_waiting_for_the_aggregator_is_shown(_coordination):
    engine = _engine()
    release = asyncio.Event()

    async def aggregator_running():
        await release.wait()

    _coordination.wait_until_aggregator_idle = aggregator_running
    engine._compute_event_plan = AsyncMock(return_value="plans")

    task = asyncio.create_task(engine.compute_event_plan(_event()))
    await asyncio.sleep(0)
    assert engine.plan_status.current.state is PlanState.WAITING
    assert not engine.plan_status.current.requested_by_user

    release.set()
    await task
    assert engine.plan_status.current.state is PlanState.IDLE


@pytest.mark.asyncio
async def test_a_failed_plan_goes_back_to_idle_with_its_error():
    engine = _engine()
    engine._compute_event_plan = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await engine.compute_event_plan(_event())

    assert engine.plan_status.current.state is PlanState.IDLE
    assert engine.plan_status.current.last_error == "boom"


@pytest.mark.asyncio
async def test_a_cancelled_plan_goes_back_to_idle():
    # Stopping the process mid-plan must not leave every UI showing a plan running.
    engine = _engine()
    started = asyncio.Event()

    async def compute(_):
        started.set()
        await asyncio.Event().wait()

    engine._compute_event_plan = compute
    task = asyncio.create_task(engine.compute_event_plan(_event()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.plan_status.current.state is PlanState.IDLE
    assert engine.plan_status.current.last_error == "CancelledError"


def test_unsubscribed_queues_get_nothing():
    tracker = PlanStatusTracker()
    with tracker.subscribe() as queue:
        pass
    tracker.waiting(_event())
    assert queue.empty()
