# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
from datetime import timedelta, datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import os
os.environ["REDISCLOUD_URL"] = "redis://mock:6379"
# Tests run without a Sight DB (no DATABASE_URL in CI), so default the Collector
# to in-process visibility. setdefault lets a dev opt into the sight path by
# exporting COLLECTOR_VISIBILITY_STRATEGY=sight with a real DATABASE_URL.
os.environ.setdefault("COLLECTOR_VISIBILITY_STRATEGY", "local")

import pytest

import astropy.units as u
from astropy.coordinates import Angle
from astropy.time import Time
from lucupy.minimodel import (ALL_SITES, CloudCover, ImageQuality, NightIndex,
                              Semester, SemesterHalf, Site, VariantSnapshot)
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
from scheduler.core.events.queue import (EndOfNightEvent, EventQueue,
                                         EveningTwilightEvent, MorningTwilightEvent,
                                         OnDemandScheduleEvent, WeatherChangeEvent)
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue
from scheduler.services.sight.database.connection import init_db_engine


@pytest.fixture(scope="session")
def visibility_calculator_fixture():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_db_engine())


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
        blueprint=collector_blueprint,
        night_times={}
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

# --- RT scaffolding (real queue, event factory, idle interlock) --------------

@pytest.fixture
def scheduler_queue():
    """A real (in-memory) SchedulerQueue, as used by the RT engine."""
    return SchedulerQueue()


class RTEventFactory:
    """Builds the event types the RT path consumes, with sensible defaults.

    Every method accepts ``site`` and ``time`` overrides so tests can shape
    scenarios without repeating boilerplate.
    """
    DEFAULT_TIME = datetime(2025, 3, 1, 23, 0, 0, tzinfo=UTC)

    def weather(self,
                site: Site = Site.GN,
                time: datetime = DEFAULT_TIME,
                iq: ImageQuality = ImageQuality.IQ70,
                cc: CloudCover = CloudCover.CC50,
                wind_dir: float = 330.0,
                wind_spd: float = 5.0) -> WeatherChangeEvent:
        variant = VariantSnapshot(iq=iq,
                                  cc=cc,
                                  wind_dir=Angle(wind_dir, unit=u.deg),
                                  wind_spd=wind_spd * (u.m / u.s))
        return WeatherChangeEvent(variant_change=variant,
                                  site=site,
                                  time=time,
                                  description=f'Weather changed for site {site.name}')

    def evening_twilight(self,
                         site: Site = Site.GN,
                         time: datetime = DEFAULT_TIME) -> EveningTwilightEvent:
        return EveningTwilightEvent(site=site, time=time,
                                    description='Evening 12 degree twilight')

    def morning_twilight(self,
                         site: Site = Site.GN,
                         time: datetime = DEFAULT_TIME) -> MorningTwilightEvent:
        return MorningTwilightEvent(site=site, time=time,
                                    description='Morning 12 degree twilight')

    def on_demand(self,
                  site: Site = Site.GN,
                  time: datetime = DEFAULT_TIME) -> OnDemandScheduleEvent:
        return OnDemandScheduleEvent(site=site, time=time,
                                     description='On-demand schedule requested')

    def end_of_night(self,
                     site: Site = Site.GN,
                     time: datetime = DEFAULT_TIME) -> EndOfNightEvent:
        return EndOfNightEvent(site=site, time=time,
                               description='End of night')


@pytest.fixture
def rt_event_factory():
    return RTEventFactory()


@pytest.fixture
def mock_coordination_idle(monkeypatch):
    """Patch the visibility-aggregator interlock so it always reports idle.

    RT tests must never wait on (or touch) the scheduler_coordination table.
    Consumers call ``coordination.<func>`` through the module attribute, so
    patching the module covers them all. Returns a namespace of AsyncMocks so
    tests can assert on interlock calls.
    """
    from scheduler.services.visibility_aggregator import coordination

    mocks = SimpleNamespace(
        wait_until_aggregator_idle=AsyncMock(return_value=True),
        is_aggregator_active=AsyncMock(return_value=False),
        is_plan_in_progress=AsyncMock(return_value=False),
        signal_plan_in_progress=AsyncMock(return_value=None),
        signal_plan_done=AsyncMock(return_value=None),
        get_aggregator_status=AsyncMock(return_value={
            'active': False, 'holder': None, 'started_at': None,
            'heartbeat_at': None, 'finished_at': None, 'stale': False,
            'detail': None,
        }),
    )
    for name, mock in vars(mocks).items():
        monkeypatch.setattr(coordination, name, mock)
    return mocks


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