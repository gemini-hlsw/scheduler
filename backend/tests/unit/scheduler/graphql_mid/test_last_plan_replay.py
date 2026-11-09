# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""A reloaded UI gets the last plan at once instead of waiting for the next one."""

import asyncio
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scheduler.core.events.queue.events import EndOfNightEvent, OnDemandScheduleEvent
from scheduler.engine.engineRT import EngineRT
from scheduler.graphql_mid.server import schema
from scheduler.graphql_mid.types import NewNightPlans, SNightTimelines, SRunSummary
from scheduler.shared_queue import last_plan_responses, plan_response_subscribers

_ID = "replay-test"
_SUB = """
subscription ($id: String!) {
  queueSchedule(scheduleId: $id) {
    __typename
    ... on NewNightPlans { plansSummary { summary } }
  }
}
"""


def _plans(tag: str) -> NewNightPlans:
    return NewNightPlans(
        night_plans=SNightTimelines(night_timeline=[]),
        plans_summary=SRunSummary(summary={"tag": tag}, metrics_per_band={}),
    )


@pytest.fixture(autouse=True)
def _clean():
    yield
    last_plan_responses.pop(_ID, None)
    plan_response_subscribers.pop(_ID, None)


async def _next_tag(sub) -> str:
    result = await asyncio.wait_for(sub.__anext__(), timeout=2)
    assert result.errors is None
    return result.data["queueSchedule"]["plansSummary"]["summary"]["tag"]


@pytest.mark.asyncio
async def test_a_new_subscriber_gets_the_last_plan_first_then_new_ones():
    last_plan_responses[_ID] = _plans("last")
    sub = await schema.subscribe(_SUB, variable_values={"id": _ID})
    try:
        assert await _next_tag(sub) == "last"

        for q in plan_response_subscribers[_ID]:
            await q.put(_plans("new"))
        assert await _next_tag(sub) == "new"
    finally:
        await sub.aclose()


@pytest.mark.asyncio
async def test_nothing_is_replayed_without_a_plan():
    sub = await schema.subscribe(_SUB, variable_values={"id": _ID})
    try:
        nxt = asyncio.ensure_future(sub.__anext__())
        await asyncio.sleep(0.05)
        assert not nxt.done()
        nxt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await nxt
    finally:
        await sub.aclose()


def _engine(results) -> EngineRT:
    """An engine whose queue hands out (event, plan) pairs, then stops the loop."""
    queue = MagicMock()
    queue.consume_events = AsyncMock(side_effect=[*results, asyncio.CancelledError()])
    store = MagicMock()
    store.reset = AsyncMock()
    return EngineRT(
        MagicMock(),
        scheduler_queue=queue,
        process_id=_ID,
        weather_source=MagicMock(),
        nightly_timeline_store=store,
    )


def _event():
    return OnDemandScheduleEvent(site=None, time=datetime.now(UTC), description="Plan", requested_by_user=True)


@pytest.mark.asyncio
async def test_the_engine_keeps_the_last_published_plan():
    plans = _plans("kept")
    await _engine([(_event(), plans), (_event(), None)]).run()
    # An event that did not plan leaves the last plan in place.
    assert last_plan_responses[_ID] is plans


@pytest.mark.asyncio
async def test_the_end_of_the_night_forgets_the_plan():
    end = EndOfNightEvent(site=None, time=datetime.now(UTC), description="End of night")
    with patch("scheduler.engine.engineRT.is_operation", True):
        await _engine([(_event(), _plans("old")), (end, None)]).run()
    assert _ID not in last_plan_responses
