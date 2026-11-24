# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Generated plans are stored as the UI receives them, without ever holding the plan up."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from unittest.mock import patch

import numpy as np
import pytest
from graphql import GraphQLList, GraphQLNonNull, GraphQLObjectType
from lucupy.minimodel import ObservationID, Site
from sqlalchemy.dialects import postgresql

from scheduler.core.builder import modes
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.events.queue.events import OnDemandScheduleEvent
from scheduler.graphql_mid.server import schema
from scheduler.graphql_mid.types import (
    NewNightPlans,
    SConditions,
    SNightInTimeline,
    SNightStats,
    SNightTimelines,
    SPlan,
    SRequiredConditions,
    SRunSummary,
    STimelineEntry,
    STimeLossWindow,
    STimestats,
    SVisit,
    TimelineEntriesBySite,
    Event as SEvent,
)
from scheduler.services import plan_archive
from scheduler.services.plan_archive import PlanArchive, PlanStatus, to_json
from scheduler.shared_queue import last_plan_responses

_START = datetime(2026, 11, 23, 23, 0, tzinfo=UTC)


def _plans() -> NewNightPlans:
    visit = SVisit(
        start_time=_START,
        end_time=_START + timedelta(hours=1),
        obs_id=ObservationID('GN-2026B-Q-1-1'),
        targets=['M31'],
        atom_start_idx=0,
        atom_end_idx=2,
        altitude=[45.0, float(np.float64(50.5))],
        instrument='GMOS-N',
        fpu='LONGSLIT',
        disperser='B600',
        filters=['g'],
        required_conditions=SRequiredConditions(iq='IQ70', cc='CC50', sb='SBANY', wv='WVANY'),
        obs_class='SCIENCE',
        score=1.5,
        peak_score=2.0,
        completion='0/3',
        atom_times=[10, 20],
        step_start=None,
        step_end=None,
        band=1,
    )
    plan = SPlan(
        site=Site.GN,
        start_time=_START,
        end_time=_START + timedelta(hours=8),
        visits=[visit],
        night_stats=SNightStats(
            plan_score=3.0, n_toos=0, completion_fraction={'1': 0.5}, program_completion={'GN-2026B-Q-1': '10%'}
        ),
        night_conditions=SConditions(iq='IQ70', cc='CC50', wv='WVANY', wind_speed=4.0, wind_direction=90.0),
    )
    entry = STimelineEntry(
        start_time_slots=0,
        event=SEvent(site=Site.GN, time=_START, description='Plan'),
        plan=plan,
        timeloss_windows=[STimeLossWindow(start=_START, end=None, loss_type='weather')],
        timestats=STimestats(night_length=480, observed=0, scheduled=60, weather=0, fault=0, closed=0, unscheduled=0),
    )
    by_site = TimelineEntriesBySite(
        site=Site.GN, time_entries=[entry], eve_twilight=_START, morn_twilight=_START + timedelta(hours=9)
    )
    return NewNightPlans(
        night_plans=SNightTimelines(night_timeline=[SNightInTimeline(night_index=0, time_entries_by_site=[by_site])]),
        plans_summary=SRunSummary(summary={'GN-2026B-Q-1': ['10%', 0.5]}, metrics_per_band={'1': {'score': 1}}),
    )


def _selection(gql_type) -> str:
    """Every field of an object type, all the way down, for a query that asks for everything."""
    while isinstance(gql_type, (GraphQLNonNull, GraphQLList)):
        gql_type = gql_type.of_type
    if not isinstance(gql_type, GraphQLObjectType):
        return ''
    return '{ ' + ' '.join(f'{name} {_selection(field.type)}' for name, field in gql_type.fields.items()) + ' }'


@pytest.mark.asyncio
async def test_a_stored_plan_is_what_the_ui_receives():
    """Compared with the queueSchedule subscription itself, so a new plan field is checked too."""
    process_id = 'plan-archive-test'
    plans = _plans()
    selection = _selection(schema._schema.type_map['NewNightPlans'])
    query = f'subscription ($id: String!) {{ queueSchedule(scheduleId: $id) {{ ... on NewNightPlans {selection} }} }}'
    last_plan_responses[process_id] = plans
    sub = await schema.subscribe(query, variable_values={'id': process_id})
    try:
        received = (await asyncio.wait_for(sub.__anext__(), timeout=2)).data['queueSchedule']
    finally:
        await sub.aclose()
        last_plan_responses.pop(process_id, None)

    assert to_json(plans) == received


def test_values_jsonb_cannot_hold_are_converted():
    assert to_json({Site.GS: [np.float64(1.5), np.int64(2), float('nan'), float('inf')]}) == {
        'GS': [1.5, 2, None, None]
    }


class _RecordingSession:
    def __init__(self, statements):
        self.statements = statements

    async def execute(self, statement):
        self.statements.append(statement)


def _recording_scope(statements, delay=0.0, fail=False):
    @asynccontextmanager
    async def session_scope():
        await asyncio.sleep(delay)
        if fail:
            raise RuntimeError('DATABASE_URL is not set')
        yield _RecordingSession(statements)

    return session_scope


def _event() -> OnDemandScheduleEvent:
    return OnDemandScheduleEvent(
        site=None, time=_START, description='Operator asked for a plan', requested_by_user=True
    )


@pytest.mark.asyncio
async def test_a_plan_is_stored_with_its_event_and_then_its_decision(monkeypatch):
    monkeypatch.setattr(modes, 'app_mode', SchedulerModes.SIMULATION)
    statements = []
    # The insert is slow on purpose: the decision must still be written after it.
    with patch.object(plan_archive, 'session_scope', _recording_scope(statements, delay=0.05)):
        archive = PlanArchive(enabled=True)
        plans = _plans()
        archive.record_plan(_event(), plans, PlanStatus.PROPOSED, 'p1')
        archive.record_decision('p1', True, 'Ada', _START)
        await archive.flush()

    insert, update = statements
    row = insert.compile(dialect=postgresql.dialect()).params
    assert row['mode'] == 'simulation'
    assert row['event_type'] == 'OnDemandScheduleEvent'
    assert row['event_description'] == 'Operator asked for a plan'
    assert row['event_time'] == _START
    assert row['event_site'] is None
    assert row['status'] == 'proposed'
    assert row['proposal_id'] == 'p1'
    assert row['plan'] == {'nightPlans': to_json(plans.night_plans), 'plansSummary': to_json(plans.plans_summary)}

    compiled = update.compile(dialect=postgresql.dialect())
    assert 'WHERE generated_plans.proposal_id' in str(compiled)
    # Only a plan still waiting can be decided on.
    assert compiled.params['status_1'] == 'proposed'
    assert {k: compiled.params[k] for k in ('status', 'decided_by', 'decided_at')} == {
        'status': 'accepted',
        'decided_by': 'Ada',
        'decided_at': _START,
    }


@pytest.mark.asyncio
async def test_a_replaced_proposal_is_marked_superseded():
    statements = []
    with patch.object(plan_archive, 'session_scope', _recording_scope(statements)):
        archive = PlanArchive(enabled=True)
        archive.record_superseded('p1')
        await archive.flush()

    (update,) = statements
    compiled = update.compile(dialect=postgresql.dialect())
    assert compiled.params == {'status': 'superseded', 'proposal_id_1': 'p1', 'status_1': 'proposed'}


@pytest.mark.asyncio
async def test_a_proposal_dropped_without_a_decision_is_marked_expired():
    statements = []
    with patch.object(plan_archive, 'session_scope', _recording_scope(statements)):
        archive = PlanArchive(enabled=True)
        archive.record_expired('p1')
        await archive.flush()

    (update,) = statements
    compiled = update.compile(dialect=postgresql.dialect())
    assert compiled.params == {'status': 'expired', 'proposal_id_1': 'p1', 'status_1': 'proposed'}


@pytest.mark.asyncio
async def test_a_database_failure_never_reaches_the_engine():
    with patch.object(plan_archive, 'session_scope', _recording_scope([], fail=True)):
        archive = PlanArchive(enabled=True)
        archive.record_plan(_event(), _plans(), PlanStatus.IN_EFFECT)
        archive.record_plan(_event(), _plans(), PlanStatus.IN_EFFECT)
        await archive.flush()


@pytest.mark.asyncio
async def test_a_disabled_archive_writes_nothing():
    statements = []
    with patch.object(plan_archive, 'session_scope', _recording_scope(statements)):
        archive = PlanArchive(enabled=False)
        archive.record_plan(_event(), _plans(), PlanStatus.IN_EFFECT)
        archive.record_decision('p1', False, 'Ada', _START)
        await archive.flush()

    assert statements == []
