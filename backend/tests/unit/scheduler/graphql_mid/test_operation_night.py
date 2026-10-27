# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
``operationNight`` reports the default night the operation plans are computed for.

It is the 12-degree twilight night derived from the clock, so it is known before the
first plan and custom night times from the build parameters do not change it.
"""

from datetime import datetime, UTC
from types import SimpleNamespace

import pytest
from lucupy.minimodel import ALL_SITES, Site

from scheduler.engine.params import default_night
from scheduler.graphql_mid import schema as schema_module

_QUERY = "{ operationNight { site start end } }"


@pytest.fixture
def operation_process(monkeypatch):
    process = SimpleNamespace(params=SimpleNamespace(sites=ALL_SITES))
    monkeypatch.setattr(schema_module.process_manager, "get_operation_process", lambda: process)
    return process


@pytest.mark.asyncio
async def test_reports_the_twilights_per_site_before_any_plan(scheduler_schema, operation_process):
    result = await scheduler_schema.execute(_QUERY)

    assert result.errors is None
    nights = result.data["operationNight"]
    assert [night["site"] for night in nights] == ["GN", "GS"]
    for night in nights:
        start, end = default_night(Site[night["site"]])
        assert datetime.fromisoformat(night["start"]) == start
        assert datetime.fromisoformat(night["end"]) == end


@pytest.mark.asyncio
async def test_only_the_sites_the_process_schedules(scheduler_schema, operation_process):
    operation_process.params.sites = frozenset([Site.GS])

    result = await scheduler_schema.execute(_QUERY)

    assert result.errors is None
    assert [night["site"] for night in result.data["operationNight"]] == ["GS"]


@pytest.mark.asyncio
async def test_empty_without_an_operation_process(scheduler_schema, monkeypatch):
    monkeypatch.setattr(schema_module.process_manager, "get_operation_process", lambda: None)

    result = await scheduler_schema.execute(_QUERY)

    assert result.errors is None
    assert result.data["operationNight"] == []


def test_default_night_is_the_twilights_of_the_target_night():
    # Mid-afternoon UT on 2026-10-27 is before GS's evening: the night ending 2026-10-28.
    start, end = default_night(Site.GS, datetime(2026, 10, 27, 18, tzinfo=UTC))

    assert (start.date(), end.date()) == (datetime(2026, 10, 27).date(), datetime(2026, 10, 28).date())
    assert start.hour == 23 and 8 <= end.hour <= 9


def test_default_night_rolls_over_after_the_night_ends():
    # 16:00 UT on 2026-10-28 is after the morning twilight at both sites, so the
    # default night is already the next one.
    now = datetime(2026, 10, 28, 16, tzinfo=UTC)
    start, end = default_night(Site.GN, now)

    assert now < start < end
