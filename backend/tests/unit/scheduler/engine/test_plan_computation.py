# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""RT-22: pure synchronous plan computation extracted from EngineRT."""

from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.time import Time
from lucupy.minimodel import Site

from scheduler.core.plans import NightStats
from scheduler.engine.plan_computation import compute_event_plans

NIGHT_START = datetime(2025, 3, 1, 23, 0, 0, tzinfo=UTC)
SITE = Site.GN


def make_scp(visits=None):
    """A mocked SCP: 1-minute timeslots, night starting at NIGHT_START."""
    scp = MagicMock()
    scp.collector.time_slot_length.to_datetime.return_value = timedelta(minutes=1)
    scp.collector.night_events = {
        SITE: MagicMock(times=[[Time(NIGHT_START.replace(tzinfo=None), scale='utc')]])
    }

    plans = MagicMock()
    plans.night_idx = 0
    plans.plans = {SITE: SimpleNamespace(visits=visits or [],
                                         night_stats=None,
                                         alt_degs=None)}
    scp.run_rt.return_value = plans
    return scp, plans


def test_event_after_twilight_starts_at_event_timeslot(rt_event_factory):
    scp, plans = make_scp()
    event = rt_event_factory.on_demand(time=NIGHT_START + timedelta(minutes=30))

    result = compute_event_plans(scp, frozenset([SITE]), event, night_times=None)

    assert result is plans, "must return the core Plans object, unconverted"
    (start_timeslots,), _ = scp.run_rt.call_args
    assert start_timeslots[SITE] == {np.int64(0): 30}


def test_event_before_twilight_starts_at_zero(rt_event_factory):
    scp, _ = make_scp()
    event = rt_event_factory.on_demand(time=NIGHT_START - timedelta(minutes=10))

    compute_event_plans(scp, frozenset([SITE]), event, night_times=None)

    (start_timeslots,), _ = scp.run_rt.call_args
    assert start_timeslots[SITE] == {np.int64(0): 0}


def test_custom_start_wins_over_earlier_event(rt_event_factory):
    """An event before the custom start begins at the custom start *timeslot*
    (not the raw Time object, which the pre-extraction code assigned)."""
    scp, _ = make_scp()
    event = rt_event_factory.on_demand(time=NIGHT_START + timedelta(minutes=30))
    custom_start = Time((NIGHT_START + timedelta(minutes=45)).replace(tzinfo=None), scale='utc')

    compute_event_plans(scp, frozenset([SITE]), event,
                        night_times={SITE: (custom_start, None)})

    (start_timeslots,), _ = scp.run_rt.call_args
    assert start_timeslots[SITE] == {np.int64(0): 45}


def test_event_after_custom_start_wins(rt_event_factory):
    scp, _ = make_scp()
    event = rt_event_factory.on_demand(time=NIGHT_START + timedelta(minutes=50))
    custom_start = Time((NIGHT_START + timedelta(minutes=45)).replace(tzinfo=None), scale='utc')

    compute_event_plans(scp, frozenset([SITE]), event,
                        night_times={SITE: (custom_start, None)})

    (start_timeslots,), _ = scp.run_rt.call_args
    assert start_timeslots[SITE] == {np.int64(0): 50}


def test_post_processing_sets_night_stats_and_alt_degs(rt_event_factory):
    visit = SimpleNamespace(obs_id='obs-1', start_time_slot=5, time_slots=3)
    scp, plans = make_scp(visits=[visit])

    # Target info: 10 altitude samples of 45d 30m 00s.
    alt_value = SimpleNamespace(dms=(45.0, 30.0, 0.0))
    scp.collector.get_target_info.return_value = {0: SimpleNamespace(alt=[alt_value] * 10)}

    event = rt_event_factory.on_demand(time=NIGHT_START + timedelta(minutes=30))
    result = compute_event_plans(scp, frozenset([SITE]), event, night_times=None)

    site_plan = result.plans[SITE]
    assert isinstance(site_plan.night_stats, NightStats)
    assert site_plan.alt_degs == [[45.5, 45.5, 45.5]]
    scp.collector.get_target_info.assert_called_once_with('obs-1')
