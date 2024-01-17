# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import pytest
from datetime import timedelta

from lucupy.minimodel import Site

from .collector_fixture import scheduler_collector


def test_utc_dt_to_time_coords(scheduler_collector):
    site = Site.GN
    timeslot_length = scheduler_collector.time_slot_length.to_datetime()
    ne = scheduler_collector.get_night_events(site)

    for night_idx in range(len(ne.time_grid)):
        # Twilights are in UTC. Convert to local time.
        start_time = ne.twilight_evening_12[night_idx].to_datetime()
        end_time = ne.twilight_morning_12[night_idx].to_datetime() - timeslot_length

        curr_time = start_time
        times = []
        while curr_time <= end_time:
            times.append(curr_time)
            curr_time += timedelta(minutes=1)

        for timeslot_idx, curr_time in enumerate(times):
            data = ne.utc_dt_to_time_coords(curr_time)
            assert data is not None
            calculated_night_idx, calculated_timeslot_idx = data
            assert night_idx == calculated_night_idx
            assert timeslot_idx == calculated_timeslot_idx


def test_local_dt_to_time_coords(scheduler_collector):
    site = Site.GN
    timeslot_length = scheduler_collector.time_slot_length.to_datetime()
    ne = scheduler_collector.get_night_events(site)

    for night_idx in range(len(ne.time_grid)):
        # Twilights are in UTC. Convert to local time.
        start_time = ne.twilight_evening_12[night_idx].to_datetime(site.timezone)
        end_time = ne.twilight_morning_12[night_idx].to_datetime(site.timezone) - timeslot_length

        curr_time = start_time
        times = []
        while curr_time <= end_time:
            times.append(curr_time)
            curr_time += timedelta(minutes=1)

        for timeslot_idx, curr_time in enumerate(times):
            data = ne.local_dt_to_time_coords(curr_time)
            assert data is not None
            calculated_night_idx, calculated_timeslot_idx = data
            assert night_idx == calculated_night_idx
            assert timeslot_idx == calculated_timeslot_idx
