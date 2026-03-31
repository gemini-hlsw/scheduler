# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import final, Dict, Tuple

from astropy.time import Time, TimeDelta
from lucupy import sky
from lucupy.minimodel import Site

from scheduler.core.calculations import NightEvents
from scheduler.core.meta import Singleton


__all__ = [
    'NightEventsManager'
]


@final
class NightEventsManager(metaclass=Singleton):
    """
    A singleton class that manages pre-calculations of NightEvents for each Site during the dates specified.
    """
    _ID = Tuple[Site, TimeDelta, Time, Time]
    _night_events: Dict[_ID, NightEvents] = {}

    @staticmethod
    def get_night_events(time_grid: Time,
                         night_start_time: Time,
                         night_end_time: Time,
                         time_slot_length: TimeDelta,
                         site: Site) -> NightEvents:
        """
        Retrieve NightEvents. These may contain more information than requested,
        but never less.
        """
        # The identifier used for caching.
        data_id: NightEventsManager._ID = (site, time_slot_length, time_grid[0], time_grid[-1])

        # Recalculate if necessary.
        if data_id not in NightEventsManager._night_events:
            night_events = NightEvents(
                time_grid,
                night_start_time,
                night_end_time,
                time_slot_length,
                site,
                *sky.night_events(time_grid, site.location, site.timezone)
            )
            NightEventsManager._night_events[data_id] = night_events

        # Check if night lenght was modified
        elif night_start_time != NightEventsManager._night_events[data_id].night_start_time or \
                night_end_time != NightEventsManager._night_events[data_id].night_end_time:
            NightEventsManager._night_events[data_id] = NightEvents(
                time_grid,
                night_start_time,
                night_end_time,
                time_slot_length,
                site,
                *sky.night_events(time_grid, site.location, site.timezone)
            )

        return NightEventsManager._night_events[data_id]
