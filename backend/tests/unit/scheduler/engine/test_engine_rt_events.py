# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""In operation, only a plan an operator asks for is computed; every other event is still recorded."""

from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import ALL_SITES, Site, VariantSnapshot

from scheduler.core.events.queue.event_history import EventHistory
from scheduler.core.events.queue.events import (
    EveningTwilightEvent,
    OnDemandScheduleEvent,
    WeatherChangeEvent,
)
from scheduler.engine.engineRT import EngineRT

_MODULE = "scheduler.engine.engineRT"
_TIME = datetime(2026, 8, 26, 6, 0, tzinfo=UTC)


def _engine() -> EngineRT:
    params = MagicMock()
    params.sites = frozenset({Site.GN})
    engine = EngineRT(
        params,
        scheduler_queue=MagicMock(),
        process_id="test",
        weather_source=MagicMock(),
        nightly_timeline_store=MagicMock(),
        event_history=EventHistory(),
    )
    engine.compute_event_plan = AsyncMock(return_value="plans")
    return engine


def _user_request() -> OnDemandScheduleEvent:
    return OnDemandScheduleEvent(site=None, time=_TIME, description="On demand request", requested_by_user=True)


def _automatic_events():
    return [
        OnDemandScheduleEvent(site=Site.GN, time=_TIME, description="Scheduler idle at GN"),
        OnDemandScheduleEvent(site=None, time=_TIME, description="Started during the night"),
        WeatherChangeEvent(
            site=Site.GS,
            time=_TIME,
            description="Weather changed for site GS",
            variant_change=MagicMock(VariantSnapshot),
        ),
        EveningTwilightEvent(site=Site.GN, time=_TIME, description="Evening 12° Twilight at GN"),
    ]


@pytest.mark.asyncio
async def test_operation_plans_for_a_user_request():
    engine = _engine()
    with patch(f"{_MODULE}.is_operation", True):
        assert await engine.handle_event(_user_request()) == "plans"
    engine.compute_event_plan.assert_awaited_once()
    (recorded,) = engine.event_history.snapshot()
    assert recorded.triggered_plan
    assert recorded.event_type == "OnDemandScheduleEvent"


@pytest.mark.asyncio
@pytest.mark.parametrize("event", _automatic_events(), ids=lambda e: type(e).__name__)
async def test_operation_records_but_does_not_plan_for_other_events(event):
    engine = _engine()
    with patch(f"{_MODULE}.is_operation", True):
        assert await engine.handle_event(event) is None
    engine.compute_event_plan.assert_not_called()
    (recorded,) = engine.event_history.snapshot()
    assert not recorded.triggered_plan
    assert recorded.event_type == type(event).__name__
    assert recorded.description == event.description
    assert recorded.sites == ((event.site,) if event.site else ())


@pytest.mark.asyncio
@pytest.mark.parametrize("event", _automatic_events(), ids=lambda e: type(e).__name__)
async def test_simulation_still_plans_for_every_event(event):
    engine = _engine()
    with patch(f"{_MODULE}.is_operation", False):
        assert await engine.handle_event(event) == "plans"


@pytest.mark.asyncio
async def test_subscribers_get_new_events_and_history_is_bounded():
    history = EventHistory(max_events=2)
    events = [OnDemandScheduleEvent(site=ALL_SITES, time=_TIME, description=f"e{i}") for i in range(3)]
    history.record(events[0], False)
    with history.subscribe() as queue:
        history.record(events[1], False)
        history.record(events[2], True)
    history.record(events[0], False)

    assert [queue.get_nowait().description for _ in range(queue.qsize())] == ["e1", "e2"]
    assert [e.description for e in history.snapshot()] == ["e2", "e0"]
    assert history.snapshot()[0].sites == tuple(sorted(ALL_SITES, key=lambda s: s.name))
