# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import final, Dict, Optional, Tuple

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
    # The custom night window is part of the identity, not just of the payload. It comes
    # from the build parameters, so it varies per run over the same dates: leaving it out
    # made two runs share one slot, and the entry was rebuilt on every alternating call
    # (see collector_refactor.md). Both are None on the ordinary twilight-to-twilight path.
    _ID = Tuple[Site, TimeDelta, Time, Time, Optional[Time], Optional[Time]]
    _night_events: Dict[_ID, NightEvents] = {}

    @staticmethod
    def get_night_events(time_grid: Time,
                         night_start_time: Optional[Time],
                         night_end_time: Optional[Time],
                         time_slot_length: TimeDelta,
                         site: Site) -> NightEvents:
        """
        Retrieve NightEvents. These may contain more information than requested,
        but never less.
        """
        # The identifier used for caching.
        data_id: NightEventsManager._ID = (site, time_slot_length, time_grid[0], time_grid[-1],
                                           night_start_time, night_end_time)

        # Recalculate if necessary.
        if data_id not in NightEventsManager._night_events:
            NightEventsManager._night_events[data_id] = NightEvents(
                time_grid,
                night_start_time,
                night_end_time,
                time_slot_length,
                site,
                *sky.night_events(time_grid, site.location, site.timezone)
            )

        return NightEventsManager._night_events[data_id]
