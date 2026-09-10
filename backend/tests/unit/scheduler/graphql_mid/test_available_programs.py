# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
``availablePrograms`` must follow the night being built.

The ODB filters programs by an active-on date, so a run of an older night has to
ask for that night; asking for today silently drops every program that has since
expired, and the operator ends up selecting from the wrong list.
"""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from lucupy.minimodel import Site

from scheduler.engine.params import BuildParameters, NightTimes, build_params_store
from scheduler.graphql_mid import schema as schema_module

_QUERY = "query($d: Date) { availablePrograms(nightDate: $d) { id refLabel } }"


@pytest.fixture
def odb_dates(monkeypatch):
    """Record the date each call sends to the ODB."""
    dates = []

    async def _get_all_reference_labels(date=None):
        dates.append(date)
        return [("G-2019A-0001", "p-1")]

    monkeypatch.setattr(
        schema_module, "gpp",
        SimpleNamespace(client=SimpleNamespace(scheduler=SimpleNamespace(
            get_all_reference_labels=_get_all_reference_labels
        )))
    )
    return dates


@pytest_asyncio.fixture(autouse=True)
async def clean_build_params():
    """The store is a singleton, so leave it as it was found."""
    yield
    await build_params_store.set(BuildParameters())


@pytest.mark.asyncio
async def test_explicit_night_date_is_forwarded(scheduler_schema, odb_dates):
    result = await scheduler_schema.execute(
        _QUERY, variable_values={"d": "2018-10-21"}
    )

    assert result.errors is None
    assert odb_dates == ["2018-10-21"]
    assert result.data["availablePrograms"][0]["refLabel"] == "G-2019A-0001"


@pytest.mark.asyncio
async def test_falls_back_to_the_build_visibility_start(scheduler_schema, odb_dates):
    await build_params_store.set(BuildParameters(
        visibility_start=datetime(2019, 3, 1, 20, tzinfo=ZoneInfo("UTC"))
    ))

    result = await scheduler_schema.execute(_QUERY, variable_values={"d": None})

    assert result.errors is None
    assert odb_dates == ["2019-03-01"]


@pytest.mark.asyncio
async def test_falls_back_to_the_earliest_night_start(scheduler_schema, odb_dates):
    await build_params_store.set(BuildParameters(night_times={
        Site.GN: NightTimes(night_start=datetime(2020, 5, 4, 6)),
        Site.GS: NightTimes(night_start=datetime(2020, 5, 2, 6)),
    }))

    result = await scheduler_schema.execute(_QUERY, variable_values={"d": None})

    assert result.errors is None
    assert odb_dates == ["2020-05-02"]


@pytest.mark.asyncio
async def test_no_build_parameters_leaves_the_date_to_the_client(
    scheduler_schema, odb_dates
):
    result = await scheduler_schema.execute(_QUERY, variable_values={"d": None})

    assert result.errors is None
    # None lets the gpp-client default to today, the old behaviour.
    assert odb_dates == [None]
