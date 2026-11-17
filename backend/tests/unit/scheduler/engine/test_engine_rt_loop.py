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
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import ObservationID, Site

from scheduler.core.events.queue.events import OnDemandScheduleEvent, OngoingObservation
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
    # Announced once the plan is written; awaited, so it cannot be a bare MagicMock.
    store.plan_published = AsyncMock()
    return store


def _engine() -> EngineRT:
    params = MagicMock()
    params.sites = _SITES
    return EngineRT(
        params,
        scheduler_queue=MagicMock(),
        process_id="test",
        weather_source=MagicMock(),
        nightly_timeline_store=_timeline_store(),
    )


def _event() -> OnDemandScheduleEvent:
    return OnDemandScheduleEvent(site=Site.GN, time=datetime(2026, 8, 26, 6, 0, tzinfo=UTC), description="test event")


def _stub_plan_pipeline(engine: EngineRT, run_rt):
    """Patch everything around run_rt so a plan can be computed without an ODB."""
    engine.build = AsyncMock()
    engine.init_variant = AsyncMock()
    engine._compute_event_start_timeslot = AsyncMock(return_value={Site.GN: {0: 0}})
    engine.scp = MagicMock()
    engine.scp.run_rt = run_rt
    return (
        patch(f"{_MODULE}.StatCalculator"),
        patch(f"{_MODULE}.SNightTimelines"),
        patch(f"{_MODULE}.SRunSummary"),
        patch(f"{_MODULE}.NewNightPlans"),
    )


@pytest.mark.asyncio
async def test_run_rt_does_not_run_on_the_loop_thread():
    ran_on = {}

    def run_rt(_start_timeslot, _ongoing):
        ran_on["thread"] = threading.get_ident()
        return MagicMock()

    engine = _engine()
    stats, timelines, summary, plans = _stub_plan_pipeline(engine, run_rt)
    with stats, timelines, summary, plans:
        await engine._compute_event_plan(_event())

    assert ran_on["thread"] != threading.get_ident(), (
        "run_rt must be offloaded; on the loop's thread it starves the ODB subscription"
    )


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

    stats, timelines, summary, plans = _stub_plan_pipeline(engine, lambda _s, _o: (time.sleep(0.3), MagicMock())[1])
    task = asyncio.create_task(ticker())
    try:
        with stats, timelines, summary, plans:
            await engine._compute_event_plan(_event())
    finally:
        task.cancel()

    assert ticks > 5, f"the loop was starved during the plan computation (ticks={ticks})"


@pytest.mark.asyncio
async def test_executing_observations_reach_the_optimizer_as_done():
    # The queue stamps what each site is executing; the plan must not schedule it again.
    received = {}

    def run_rt(_start_timeslot, ongoing):
        received["ongoing"] = ongoing
        return MagicMock()

    obs_id = ObservationID("G-2026B-0001-Q-0001")
    end = datetime(2026, 8, 26, 6, 40, tzinfo=UTC)
    event = OnDemandScheduleEvent(
        site=Site.GN,
        time=datetime(2026, 8, 26, 6, 0, tzinfo=UTC),
        description="test",
        ongoing_observations={Site.GN: OngoingObservation(obs_id=obs_id, start=_NIGHT_START, expected_end=end)},
    )
    engine = _engine()
    stats, timelines, summary, plans = _stub_plan_pipeline(engine, run_rt)
    with stats, timelines, summary, plans:
        await engine._compute_event_plan(event)

    assert received["ongoing"] == {obs_id: end}


_NIGHT_START = datetime(2026, 8, 26, 5, 0, tzinfo=UTC)


def _engine_with_night() -> EngineRT:
    """An engine whose GN night starts at 05:00 with 1-minute time slots."""
    engine = _engine()
    engine.scp = MagicMock()
    engine.scp.collector.get_night_length.return_value = (_NIGHT_START, _NIGHT_START + timedelta(hours=10))
    engine.scp.collector.get_twilights.return_value = (_NIGHT_START, _NIGHT_START + timedelta(hours=10))
    engine.scp.collector.get_custom_night_times.return_value = (None, None)
    engine.scp.collector.time_slot_length.to_datetime.return_value = timedelta(minutes=1)
    return engine


def _event_at(minutes_into_night: int, expected_end_minutes=None) -> OnDemandScheduleEvent:
    ongoing = {}
    if expected_end_minutes is not None:
        ongoing = {
            Site.GN: OngoingObservation(
                obs_id=ObservationID("G-2026B-0001-Q-0001"),
                start=_NIGHT_START,
                expected_end=_NIGHT_START + timedelta(minutes=expected_end_minutes),
            )
        }
    return OnDemandScheduleEvent(
        site=Site.GN,
        time=_NIGHT_START + timedelta(minutes=minutes_into_night),
        description="test",
        ongoing_observations=ongoing,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event,expected_slot",
    [
        # Nothing running: the plan starts when the event happened.
        (_event_at(60), 60),
        # Running until 01:40 into the night: the plan starts once it is expected to finish.
        (_event_at(60, expected_end_minutes=100), 100),
        # Already overran its estimate: nothing to wait for.
        (_event_at(60, expected_end_minutes=50), 60),
    ],
)
async def test_plan_starts_once_the_executing_observation_should_finish(event, expected_slot):
    start_timeslot = await _engine_with_night()._compute_event_start_timeslot(event)

    assert start_timeslot[Site.GN][0] == expected_slot


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
