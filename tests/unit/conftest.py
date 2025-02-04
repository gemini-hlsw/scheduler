# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
import pytest

from astropy.time import Time
from lucupy.minimodel import ALL_SITES, NightIndex, Semester, SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.graphql_mid.server import schema
from scheduler.core.builder.blueprint import CollectorBlueprint
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.sources.sources import Sources
from scheduler.core.eventsqueue import EventQueue
from scheduler.services.visibility import visibility_calculator


@pytest.fixture(scope="session")
def visibility_calculator_fixture():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(visibility_calculator.calculate())


@pytest.fixture(scope="module")
def scheduler_collector():
    start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
    num_nights_to_schedule = 3
    sites = ALL_SITES

    semesters = frozenset([Semester(2018, SemesterHalf.B)])
    collector_blueprint = CollectorBlueprint(
        obs_class=['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        prg_type=['Q', 'LP', 'FT', 'DD'],
        time_slot_length=1.0
    )

    night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))
    builder = ValidationBuilder(Sources(), EventQueue(night_indices, sites))

    collector = builder.build_collector(
        start=start,
        end=end,
        num_of_nights=num_nights_to_schedule,
        sites=sites,
        semesters=semesters,
        with_redis=True,
        blueprint=collector_blueprint
    )

    return collector


@pytest.fixture(scope="session")
def set_observatory_properties():
    ObservatoryProperties.set_properties(GeminiProperties)


@pytest.fixture(scope="session")
def scheduler_schema():
    return schema