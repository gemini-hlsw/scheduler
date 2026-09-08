# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Plan computation must not block the event loop.

The Night Monitor's ODB subscription lives on the same loop as the engine, and its
websocket gives up after 10s. A synchronous plan therefore dropped the subscription on
every replan -- and the events emitted while it was down are gone, because the
obscalcUpdate subscription has no cursor to resume from.
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import Site

from scheduler.core.events.queue.events import OnDemandScheduleEvent
from scheduler.engine.engineRT import EngineRT

_MODULE = "scheduler.engine.engineRT"
_SITES = frozenset({Site.GN})


@pytest.fixture(autouse=True)
def _quiet_coordination():
    """The aggregator interlock is not what these tests are about."""
    with patch(f"{_MODULE}.coordination") as coord:
        coord.wait_until_aggregator_idle = AsyncMock()
        coord.signal_plan_in_progress = AsyncMock()
        coord.signal_plan_done = AsyncMock()
        yield coord


def _timeline_store() -> MagicMock:
    """A store whose mutate() yields a plain (non-async) timeline.

    A bare MagicMock would yield an AsyncMock, turning the synchronous timeline writes
    into un-awaited coroutines and hiding real mistakes behind warnings.
    """
    @asynccontextmanager
    async def mutate():
        yield MagicMock()

    store = MagicMock()
    store.mutate = mutate
    return store


def _engine() -> EngineRT:
    params = MagicMock()
    params.sites = _SITES
    return EngineRT(params,
                    scheduler_queue=MagicMock(),
                    process_id="test",
                    weather_source=MagicMock(),
                    nightly_timeline_store=_timeline_store())


def _event() -> OnDemandScheduleEvent:
    return OnDemandScheduleEvent(site=Site.GN,
                                 time=datetime(2026, 8, 26, 6, 0, tzinfo=UTC),
                                 description="test event")


def _stub_plan_pipeline(engine: EngineRT, run_rt):
    """Patch everything around run_rt so a plan can be computed without an ODB."""
    engine.build = AsyncMock()
    engine.init_variant = AsyncMock()
    engine._compute_event_start_timeslot = AsyncMock(return_value={Site.GN: {0: 0}})
    engine.scp = MagicMock()
    engine.scp.run_rt = run_rt
    return (patch(f"{_MODULE}.StatCalculator"), patch(f"{_MODULE}.SNightTimelines"),
            patch(f"{_MODULE}.SRunSummary"), patch(f"{_MODULE}.NewNightPlans"))


@pytest.mark.asyncio
async def test_run_rt_does_not_run_on_the_loop_thread():
    ran_on = {}

    def run_rt(_start_timeslot):
        ran_on["thread"] = threading.get_ident()
        return MagicMock()

    engine = _engine()
    stats, timelines, summary, plans = _stub_plan_pipeline(engine, run_rt)
    with stats, timelines, summary, plans:
        await engine._compute_event_plan(_event())

    assert ran_on["thread"] != threading.get_ident(), \
        "run_rt must be offloaded; on the loop's thread it starves the ODB subscription"


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_a_plan_computes():
    """The regression that matters: a blocking plan must not stall other tasks.

    A 0.3s block stands in for a real plan. The ticker counts how many times the loop got
    scheduled; inline, it would get exactly zero.
    """
    engine = _engine()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    stats, timelines, summary, plans = _stub_plan_pipeline(
        engine, lambda _s: (time.sleep(0.3), MagicMock())[1]
    )
    task = asyncio.create_task(ticker())
    try:
        with stats, timelines, summary, plans:
            await engine._compute_event_plan(_event())
    finally:
        task.cancel()

    assert ticks > 5, f"the loop was starved during the plan computation (ticks={ticks})"


@pytest.mark.asyncio
async def test_a_plan_logs_its_duration():
    # Without this there is no way to notice plans getting slower until something
    # downstream times out.
    engine = _engine()
    engine._compute_event_plan = AsyncMock(return_value="plans")

    with patch(f"{_MODULE}._logger") as logger:
        assert await engine.compute_event_plan(_event()) == "plans"

    assert "computed in" in " ".join(str(c) for c in logger.info.call_args_list)


@pytest.mark.asyncio
async def test_a_failed_plan_still_releases_the_interlock(_quiet_coordination):
    # Otherwise the visibility aggregator blocks forever on a plan that already died.
    engine = _engine()
    engine._compute_event_plan = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await engine.compute_event_plan(_event())

    _quiet_coordination.signal_plan_done.assert_awaited_once()
