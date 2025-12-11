# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
from datetime import timedelta, datetime
from unittest.mock import MagicMock

import pytest

from astropy.time import Time
from lucupy.minimodel import ALL_SITES, NightIndex, Semester, SemesterHalf, Site
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.core.calculations import NightEvents
from scheduler.core.components.changemonitor import ChangeMonitor
from scheduler.core.events.cycle import EventCycle
from scheduler.core.scp import SCP
from scheduler.graphql_mid.server import schema
from scheduler.core.builder.blueprint import CollectorBlueprint
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.sources.sources import Sources
from scheduler.core.events.queue import EventQueue
from scheduler.services.visibility import visibility_calculator


@pytest.fixture(scope="session")
def visibility_calculator_fixture():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(visibility_calculator.calculate())


@pytest.fixture(scope="module")
def scheduler_collector():
    start = datetime.fromisoformat("2018-10-01 08:00:00")
    end = datetime.fromisoformat("2018-10-03 08:00:00")
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
        blueprint=collector_blueprint
    )

    return collector


@pytest.fixture(scope="session")
def set_observatory_properties():
    ObservatoryProperties.set_properties(GeminiProperties)


@pytest.fixture(scope="session")
def scheduler_schema():
    return schema

@pytest.fixture
def setup_basic_components():
    """Set up basic components needed for all tests."""
    params = MagicMock()
    params.sites = Site.GN

    scp = MagicMock(spec=SCP)
    scp.collector = MagicMock()
    scp.collector.time_slot_length = MagicMock()
    scp.collector.time_slot_length.to_datetime.return_value = timedelta(minutes=15)
    scp.selector = MagicMock()

    queue = MagicMock(spec=EventQueue)

    return {
        'params': params,
        'scp': scp,
        'queue': queue
    }

@pytest.fixture
def setup_event_cycle(setup_basic_components):
    """Create an EventCycle instance with mocked dependencies."""
    comps = setup_basic_components

    # Create EventCycle
    event_cycle = EventCycle(
        params=comps['params'],
        queue=comps['queue'],
        scp=comps['scp']
    )

    # Mock the change monitor
    event_cycle.change_monitor = MagicMock(spec=ChangeMonitor)
    event_cycle.change_monitor.is_site_unblocked.return_value = True

    return event_cycle, comps

@pytest.fixture
def setup_night_events():
    """Set up mock night events."""
    # Base time for testing
    base_time = datetime(2018, 9, 18, 18, 0, 0)

    # Create night events with twilight times
    night_events = MagicMock(spec=NightEvents)
    evening_twilight = MagicMock()
    evening_twilight.to_datetime.return_value = base_time
    night_events.twilight_evening_12 = {NightIndex(0): evening_twilight}

    return {
        'night_events': night_events,
        'base_time': base_time
    }