# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from types import SimpleNamespace

import pytest
from lucupy.minimodel import NightIndex, ObservationID, Site, TimeslotIndex

from scheduler.core.events.queue import NightlyTimeline, NightlyTimelineStore, TimelineEntry, TimeStats

NIGHT_IDX = NightIndex(0)


def fake_plan(*obs_ids):
    """A stand-in for Plan: the store only ever reads visits and deep-copies the whole thing."""
    return SimpleNamespace(visits=[SimpleNamespace(observation=SimpleNamespace(id=ObservationID(obs_id))) for obs_id in obs_ids])


def entry(plan):
    return TimelineEntry(start_time_slot=TimeslotIndex(0),
                         event=SimpleNamespace(description='test event'),
                         plan_generated=plan,
                         accounted_observations=[],
                         timeloss_windows=[],
                         timestats=TimeStats(0, 0, 0, 0, 0, 0, 0))


def store_with(site=Site.GN, timeline_plans=(), stitched_plans=()):
    """A store whose timelines are populated directly, bypassing the stitching logic."""
    timeline = NightlyTimeline()
    if timeline_plans:
        timeline.timeline[NIGHT_IDX] = {site: [entry(p) for p in timeline_plans]}
    if stitched_plans:
        timeline.stitched_timeline[NIGHT_IDX] = {site: [entry(p) for p in stitched_plans]}
    return NightlyTimelineStore(timeline)


@pytest.mark.asyncio
async def test_mutate_exposes_the_live_timeline():
    """Writers see the same object across calls, so their edits accumulate."""
    store = NightlyTimelineStore()

    async with store.mutate() as timeline:
        timeline.timeline[NIGHT_IDX] = {Site.GN: [entry(fake_plan('GN-1'))]}

    async with store.mutate() as same_timeline:
        assert same_timeline.timeline[NIGHT_IDX][Site.GN][0].plan_generated.visits[0].observation.id == ObservationID('GN-1')


@pytest.mark.asyncio
async def test_mutate_serializes_concurrent_writers():
    """The second block must wait for the first to finish, not interleave with it."""
    store = NightlyTimelineStore()
    order = []

    async def writer(name):
        async with store.mutate():
            order.append(f'{name}-start')
            await asyncio.sleep(0.01)
            order.append(f'{name}-end')

    await asyncio.gather(writer('a'), writer('b'))

    assert order in (['a-start', 'a-end', 'b-start', 'b-end'],
                     ['b-start', 'b-end', 'a-start', 'a-end'])


@pytest.mark.asyncio
async def test_last_plan_prefers_the_stitched_timeline():
    """The plan in effect is the stitched one: the raw entry only covers the last event."""
    store = store_with(timeline_plans=[fake_plan('GN-raw')], stitched_plans=[fake_plan('GN-stitched')])

    plan = await store.last_plan(Site.GN)

    assert [v.observation.id for v in plan.visits] == [ObservationID('GN-stitched')]


@pytest.mark.asyncio
async def test_last_plan_falls_back_to_the_raw_timeline():
    """The first plan of the night is recorded before anything is stitched."""
    store = store_with(timeline_plans=[fake_plan('GN-raw')])

    plan = await store.last_plan(Site.GN)

    assert [v.observation.id for v in plan.visits] == [ObservationID('GN-raw')]


@pytest.mark.asyncio
async def test_last_plan_skips_entries_without_a_plan():
    """Events like faults are recorded with no plan; the last plan is the last one that exists."""
    store = store_with(stitched_plans=[fake_plan('GN-1'), None])

    plan = await store.last_plan(Site.GN)

    assert [v.observation.id for v in plan.visits] == [ObservationID('GN-1')]


@pytest.mark.asyncio
async def test_last_plan_returns_a_copy():
    """A reader mutating what it got back must not corrupt the engine's timeline."""
    store = store_with(stitched_plans=[fake_plan('GN-1')])

    plan = await store.last_plan(Site.GN)
    plan.visits.clear()

    assert len((await store.last_plan(Site.GN)).visits) == 1


@pytest.mark.asyncio
async def test_reads_on_an_empty_store_are_a_no_op_path():
    """Handlers can fire before the first plan exists, or for a site with no plan."""
    store = NightlyTimelineStore()

    assert await store.last_plan(Site.GN) is None
    assert await store.planned_observation_ids(Site.GN) == frozenset()
    assert await store.has_plan(Site.GN) is False

    populated = store_with(site=Site.GN, stitched_plans=[fake_plan('GN-1')])
    assert await populated.last_plan(Site.GS) is None
    assert await populated.has_plan(Site.GS) is False


@pytest.mark.asyncio
async def test_planned_observation_ids_and_has_plan():
    store = store_with(stitched_plans=[fake_plan('GN-1', 'GN-2')])

    assert await store.planned_observation_ids(Site.GN) == frozenset(
        {ObservationID('GN-1'), ObservationID('GN-2')})
    assert await store.has_plan(Site.GN) is True


@pytest.mark.asyncio
async def test_reset_clears_the_timeline():
    store = store_with(stitched_plans=[fake_plan('GN-1')])

    await store.reset()

    assert await store.last_plan(Site.GN) is None


@pytest.mark.asyncio
async def test_engine_clears_the_store_at_end_of_night():
    """The night's timeline must not survive as the plan in effect for whatever runs next."""
    from unittest.mock import AsyncMock, MagicMock

    from scheduler.core.events.queue.events import EndOfNightEvent
    from scheduler.engine.engineRT import EngineRT

    store = store_with(stitched_plans=[fake_plan('GN-1')])
    end_of_night = EndOfNightEvent(site=Site.GN, time=None, description='End of Night')

    engine = EngineRT(MagicMock(), MagicMock(), 'test', MagicMock(), store)
    engine.scheduler_queue.consume_events = AsyncMock(return_value=(end_of_night, None))

    await engine.run()

    assert await store.last_plan(Site.GN) is None
