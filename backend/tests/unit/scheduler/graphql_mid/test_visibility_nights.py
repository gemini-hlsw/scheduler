# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
``visibilityNights`` shortens an operation build's visibility period to that many
days past tonight, instead of running to the semester end.

It is a count rather than dates so the server keeps deciding which night is
tonight: dates sent from the UI would pin the night and stop it rolling over.
"""

from datetime import datetime, UTC

import pytest
import pytest_asyncio
from pydantic import ValidationError

from scheduler.engine.params import BuildParameters, build_params_store

_START = datetime(2026, 10, 28, 8, tzinfo=UTC)
_SEMESTER_END = datetime(2027, 2, 1, 8, tzinfo=UTC)

_MUTATION = "mutation($n: Int) { updateBuildParams(buildParamsInput: { visibilityNights: $n }) }"
_QUERY = "{ buildParameters { visibilityNights visibilityEnd } }"


@pytest_asyncio.fixture(autouse=True)
async def clean_build_params():
    """The store is a singleton, so leave it as it was found."""
    yield
    await build_params_store.set(BuildParameters())


def test_default_runs_to_the_semester_end():
    assert BuildParameters().resolve_visibility_end(_START, _SEMESTER_END) == _SEMESTER_END


def test_nights_count_from_the_start():
    params = BuildParameters(visibility_nights=10)

    assert params.resolve_visibility_end(_START, _SEMESTER_END) == datetime(2026, 11, 7, 8, tzinfo=UTC)


def test_nights_stop_at_the_semester_end():
    params = BuildParameters(visibility_nights=500)

    assert params.resolve_visibility_end(_START, _SEMESTER_END) == _SEMESTER_END


def test_an_explicit_end_wins_over_nights():
    end = datetime(2026, 11, 1, 8, tzinfo=UTC)
    params = BuildParameters(visibility_end=end, visibility_nights=10)

    assert params.resolve_visibility_end(_START, _SEMESTER_END) == end


def test_nights_do_not_customize_the_night():
    assert not BuildParameters(visibility_nights=10).is_customized()


def test_zero_nights_is_refused():
    with pytest.raises(ValidationError):
        BuildParameters(visibility_nights=0)


@pytest.mark.asyncio
async def test_nights_round_trip_through_the_api(scheduler_schema):
    result = await scheduler_schema.execute(_MUTATION, variable_values={"n": 10})
    assert result.errors is None
    assert "successful" in result.data["updateBuildParams"]

    result = await scheduler_schema.execute(_QUERY)

    assert result.errors is None
    assert result.data["buildParameters"] == {"visibilityNights": 10, "visibilityEnd": None}


@pytest.mark.asyncio
async def test_api_refuses_zero_nights(scheduler_schema):
    result = await scheduler_schema.execute(_MUTATION, variable_values={"n": 0})

    assert result.errors is None
    assert result.data["updateBuildParams"].startswith("Error")
    assert (await build_params_store.get()).visibility_nights is None
