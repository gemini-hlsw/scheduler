# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""A plan computed while another is in effect is only proposed until someone accepts it."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import Site

from scheduler.core.events.queue.events import OnDemandScheduleEvent
from scheduler.engine.engineRT import EngineRT, PlanProposal, StaleProposalError
from scheduler.graphql_mid.server import schema
from scheduler.graphql_mid.types import (
    NewNightPlans,
    PlanDecision,
    ProposedNightPlans,
    SNightTimelines,
    SRunSummary,
)
from scheduler.shared_queue import last_plan_responses, pending_plan_responses, plan_response_subscribers

_MODULE = "scheduler.engine.engineRT"
_ID = "proposal-test"


@pytest.fixture(autouse=True)
def _clean():
    yield
    for registry in (last_plan_responses, pending_plan_responses, plan_response_subscribers):
        registry.pop(_ID, None)


def _summary(tag: str) -> SRunSummary:
    return SRunSummary(summary={"tag": tag}, metrics_per_band={})


def _store(in_effect: bool) -> MagicMock:
    live, copy = MagicMock(name="live"), MagicMock(name="copy")

    @asynccontextmanager
    async def mutate():
        yield live

    store = MagicMock()
    store.mutate = mutate
    store.live, store.copy = live, copy
    store.plan_in_effect.return_value = in_effect
    store.copy_timeline.return_value = copy
    store.plan_published = AsyncMock()
    store.replace = AsyncMock()
    return store


def _engine(store: MagicMock) -> EngineRT:
    params = MagicMock()
    params.sites = [Site.GN]
    engine = EngineRT(
        params, scheduler_queue=MagicMock(), process_id=_ID, weather_source=MagicMock(), nightly_timeline_store=store
    )
    engine.build = AsyncMock()
    engine.clear_timelines_if_needed = AsyncMock()
    engine.init_variant = AsyncMock()
    engine._compute_event_start_timeslot = AsyncMock(return_value={Site.GN: {0: 0}})
    engine._drop_pseudo_time_accounting = MagicMock()
    engine.scp = MagicMock()
    engine.scp.collector.get_observation_ids.return_value = []
    return engine


def _event() -> OnDemandScheduleEvent:
    return OnDemandScheduleEvent(site=None, time=datetime.now(UTC), description="Plan", requested_by_user=True)


async def _compute(engine: EngineRT):
    with (
        patch(f"{_MODULE}.StatCalculator"),
        patch(
            f"{_MODULE}.SNightTimelines.from_computed_stitched_timelines",
            return_value=SNightTimelines(night_timeline=[]),
        ),
        patch(f"{_MODULE}.SRunSummary.from_computed_run_summary", return_value=_summary("computed")),
    ):
        return await engine._compute_event_plan(_event())


@pytest.mark.asyncio
async def test_the_first_plan_of_the_night_takes_effect_at_once():
    store = _store(in_effect=False)
    engine = _engine(store)

    plans = await _compute(engine)

    assert isinstance(plans, NewNightPlans)
    store.copy_timeline.assert_not_called()
    store.live.add.assert_called_once()
    store.plan_published.assert_awaited_once_with(Site.GN)
    assert engine.proposal is None


@pytest.mark.asyncio
async def test_a_later_plan_is_only_proposed():
    store = _store(in_effect=True)
    engine = _engine(store)

    plans = await _compute(engine)

    assert isinstance(plans, ProposedNightPlans)
    # Stitched onto the copy: the plan in effect, and what the Night Monitor watches, is untouched.
    store.copy.add.assert_called_once()
    store.live.add.assert_not_called()
    store.set_collected_observation_ids.assert_not_called()
    store.plan_published.assert_not_awaited()
    assert engine.proposal.proposal_id == plans.proposal_id
    assert pending_plan_responses[_ID] is plans


def _proposed(engine: EngineRT, store: MagicMock, proposal_id: str = "p1") -> ProposedNightPlans:
    response = ProposedNightPlans(
        proposal_id=proposal_id, night_plans=SNightTimelines(night_timeline=[]), plans_summary=_summary("proposed")
    )
    engine.proposal = PlanProposal(
        proposal_id=proposal_id, timeline=store.copy, collected_ids=frozenset(), sites=[Site.GN], response=response
    )
    pending_plan_responses[_ID] = response
    return response


def _listen() -> asyncio.Queue:
    queue = asyncio.Queue()
    plan_response_subscribers.setdefault(_ID, set()).add(queue)
    return queue


@pytest.mark.asyncio
async def test_accepting_puts_the_proposal_in_effect_for_everyone():
    store = _store(in_effect=True)
    engine = _engine(store)
    _proposed(engine, store)
    ui = _listen()

    await engine.decide_plan("p1", accept=True, decided_by="Ada")

    store.replace.assert_awaited_once_with(store.copy, frozenset())
    store.plan_published.assert_awaited_once_with(Site.GN)
    assert last_plan_responses[_ID].plans_summary.summary == {"tag": "proposed"}
    assert _ID not in pending_plan_responses
    assert ui.get_nowait() == PlanDecision(proposal_id="p1", accepted=True, decided_by="Ada")


@pytest.mark.asyncio
async def test_discarding_keeps_the_plan_in_effect():
    store = _store(in_effect=True)
    engine = _engine(store)
    kept = NewNightPlans(night_plans=SNightTimelines(night_timeline=[]), plans_summary=_summary("kept"))
    last_plan_responses[_ID] = kept
    _proposed(engine, store)
    ui = _listen()

    await engine.decide_plan("p1", accept=False, decided_by="Ada")

    store.replace.assert_not_awaited()
    store.plan_published.assert_not_awaited()
    assert last_plan_responses[_ID] is kept
    assert _ID not in pending_plan_responses
    assert ui.get_nowait().accepted is False


@pytest.mark.asyncio
async def test_a_plan_is_decided_only_once():
    store = _store(in_effect=True)
    engine = _engine(store)
    _proposed(engine, store)

    await engine.decide_plan("p1", accept=False, decided_by="Ada")
    with pytest.raises(StaleProposalError):
        await engine.decide_plan("p1", accept=True, decided_by="Grace")
    store.replace.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_replaced_proposal_cannot_be_decided():
    store = _store(in_effect=True)
    engine = _engine(store)
    _proposed(engine, store, "old")
    _proposed(engine, store, "new")

    with pytest.raises(StaleProposalError):
        await engine.decide_plan("old", accept=True, decided_by="Ada")
    assert engine.proposal.proposal_id == "new"


@pytest.mark.asyncio
async def test_a_failed_plan_drops_the_proposal():
    store = _store(in_effect=True)
    engine = _engine(store)
    _proposed(engine, store)
    queue = MagicMock()
    queue.consume_events = AsyncMock(side_effect=[RuntimeError("boom"), asyncio.CancelledError()])
    engine.scheduler_queue = queue

    await engine.run()

    assert engine.proposal is None
    assert _ID not in pending_plan_responses


_SUB = """
subscription ($id: String!) {
  queueSchedule(scheduleId: $id) {
    __typename
    ... on NewNightPlans { plansSummary { summary } }
    ... on ProposedNightPlans { proposalId plansSummary { summary } }
  }
}
"""


@pytest.mark.asyncio
async def test_a_reloaded_ui_gets_the_plan_in_effect_then_the_proposal():
    last_plan_responses[_ID] = NewNightPlans(
        night_plans=SNightTimelines(night_timeline=[]), plans_summary=_summary("in effect")
    )
    pending_plan_responses[_ID] = ProposedNightPlans(
        proposal_id="p1", night_plans=SNightTimelines(night_timeline=[]), plans_summary=_summary("proposed")
    )
    sub = await schema.subscribe(_SUB, variable_values={"id": _ID})
    try:
        first = (await asyncio.wait_for(sub.__anext__(), timeout=2)).data["queueSchedule"]
        second = (await asyncio.wait_for(sub.__anext__(), timeout=2)).data["queueSchedule"]
    finally:
        await sub.aclose()

    assert first["__typename"] == "NewNightPlans"
    assert second == {
        "__typename": "ProposedNightPlans",
        "proposalId": "p1",
        "plansSummary": {"summary": {"tag": "proposed"}},
    }


_DECIDE = """
mutation ($id: String!, $accept: Boolean!) {
  decidePlan(proposalId: $id, accept: $accept) { proposalId accepted decidedBy }
}
"""

_SUB_ALL = """
subscription ($id: String!) {
  queueSchedule(scheduleId: $id) {
    __typename
    ... on PlanDecision { proposalId accepted decidedBy }
  }
}
"""


@pytest.mark.asyncio
async def test_one_ui_decides_and_every_ui_is_told():
    store = _store(in_effect=True)
    engine = _engine(store)
    _proposed(engine, store)
    process = MagicMock(engine=engine)
    first = await schema.subscribe(_SUB_ALL, variable_values={"id": _ID})
    second = await schema.subscribe(_SUB_ALL, variable_values={"id": _ID})
    try:
        # Both get the proposal replayed, so both are subscribed before the decision.
        for sub in (first, second):
            replay = await asyncio.wait_for(sub.__anext__(), timeout=2)
            assert replay.data["queueSchedule"]["__typename"] == "ProposedNightPlans"

        with patch("scheduler.graphql_mid.schema.process_manager.get_operation_process", return_value=process):
            result = await schema.execute(_DECIDE, variable_values={"id": "p1", "accept": True})
            stale = await schema.execute(_DECIDE, variable_values={"id": "p1", "accept": False})

        assert result.errors is None
        assert result.data["decidePlan"] == {"proposalId": "p1", "accepted": True, "decidedBy": "anonymous"}
        assert "no longer waiting" in stale.errors[0].message
        for sub in (first, second):
            told = await asyncio.wait_for(sub.__anext__(), timeout=2)
            assert told.data["queueSchedule"] == {
                "__typename": "PlanDecision",
                "proposalId": "p1",
                "accepted": True,
                "decidedBy": "anonymous",
            }
    finally:
        await first.aclose()
        await second.aclose()

    # A UI loaded afterwards gets the accepted plan and no proposal.
    assert _ID not in pending_plan_responses
    assert last_plan_responses[_ID].plans_summary.summary == {"tag": "proposed"}
