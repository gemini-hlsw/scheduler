# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The plan history queries the UI's Plan History tab reads stored plans with."""

from contextlib import asynccontextmanager
from datetime import date, datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scheduler.graphql_mid.server import schema
from scheduler.services.plan_history import PlanNight

_SCHEMA = 'scheduler.graphql_mid.schema'


@asynccontextmanager
async def _no_session():
    yield object()


def _row(**overrides):
    row = dict(
        id=7,
        created_at=datetime(2026, 11, 24, 2, 0, tzinfo=UTC),
        mode='operation',
        app_version='2026.11.5-staging-realtime',
        night_date=date(2026, 11, 24),
        event_type='OnDemandScheduleEvent',
        event_description='New schedule requested by Ada',
        event_time=datetime(2026, 11, 24, 1, 59, tzinfo=UTC),
        event_site=None,
        status='discarded',
        proposal_id='p1',
        decided_by='Grace',
        decided_at=datetime(2026, 11, 24, 2, 5, tzinfo=UTC),
    )
    row.update(overrides)
    return SimpleNamespace(**row)


@pytest.mark.asyncio
async def test_nights_with_plans():
    nights = AsyncMock(return_value=[PlanNight(date(2026, 11, 24), 5), PlanNight(date(2026, 11, 23), 2)])
    with patch(f'{_SCHEMA}.session_scope', _no_session), patch(f'{_SCHEMA}.plan_history.list_plan_nights', nights):
        result = await schema.execute('query { generatedPlanNights(mode: "simulation") { nightDate planCount } }')

    assert result.errors is None
    assert result.data['generatedPlanNights'] == [
        {'nightDate': '2026-11-24', 'planCount': 5},
        {'nightDate': '2026-11-23', 'planCount': 2},
    ]
    assert nights.await_args.kwargs['mode'] == 'simulation'


@pytest.mark.asyncio
async def test_a_nights_plans_with_their_status():
    plans = AsyncMock(return_value=[_row(), _row(id=8, status='in_effect', proposal_id=None, decided_by=None)])
    with patch(f'{_SCHEMA}.session_scope', _no_session), patch(f'{_SCHEMA}.plan_history.list_night_plans', plans):
        result = await schema.execute(
            'query { generatedPlans(nightDate: "2026-11-24") { id mode eventDescription status decidedBy } }'
        )

    assert result.errors is None
    assert result.data['generatedPlans'] == [
        {
            'id': 7,
            'mode': 'operation',
            'eventDescription': 'New schedule requested by Ada',
            'status': 'DISCARDED',
            'decidedBy': 'Grace',
        },
        {
            'id': 8,
            'mode': 'operation',
            'eventDescription': 'New schedule requested by Ada',
            'status': 'IN_EFFECT',
            'decidedBy': None,
        },
    ]
    assert plans.await_args.args[1] == date(2026, 11, 24)


@pytest.mark.asyncio
async def test_a_plans_content_is_returned_as_stored():
    stored = {'nightPlans': {'nightTimeline': []}, 'plansSummary': {'summary': {}, 'metricsPerBand': {}}}
    content = AsyncMock(side_effect=[stored, None])
    with patch(f'{_SCHEMA}.session_scope', _no_session), patch(f'{_SCHEMA}.plan_history.get_plan_content', content):
        found = await schema.execute('query { generatedPlanContent(planId: 7) }')
        missing = await schema.execute('query { generatedPlanContent(planId: 999) }')

    assert found.data['generatedPlanContent'] == stored
    assert missing.errors is None and missing.data['generatedPlanContent'] is None
