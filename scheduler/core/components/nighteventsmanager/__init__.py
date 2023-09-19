# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import time
from typing import Dict, Tuple

from astropy.time import Time, TimeDelta
from lucupy import sky
from lucupy.minimodel import Site

from scheduler.core.calculations import NightEvents
from scheduler.core.meta import Singleton


class NightEventsManager(metaclass=Singleton):
    """
    A singleton class that manages pre-calculations of NightEvents for each Site during the dates specified.
    """
    _data_id = Tuple[Site, TimeDelta, Time, Time]
    _night_events: Dict[_data_id, NightEvents] = {}

    @staticmethod
    def get_night_events(time_grid: Time,
                         time_slot_length: TimeDelta,
                         site: Site) -> NightEvents:
        """
        Retrieve NightEvents. These may contain more information than requested,
        but never less.
        """
        start = time.perf_counter()
        # The identifier used for caching.
        data_id = (site, time_slot_length.jd, time_grid[0].jd, time_grid[-1].jd)

        # Recalculate if we are not compatible.
        if data_id not in NightEventsManager._night_events:
            night_events = NightEvents(
                time_grid,
                time_slot_length,
                site,
                *sky.night_events(time_grid, site.location, site.timezone)
            )
            NightEventsManager._night_events[data_id] = night_events

        end = time.perf_counter()
        print(f"*** get_night_events took {end - start:.8f} seconds")

        return NightEventsManager._night_events[data_id]
