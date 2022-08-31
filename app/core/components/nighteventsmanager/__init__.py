# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Dict

from astropy.time import Time, TimeDelta
from lucupy import sky
from lucupy.minimodel import Site

from app.core.calculations import NightEvents
from app.core.components.base import SchedulerComponent


class NightEventsManager(SchedulerComponent):
    """
    Manages pre-calculation of NightEvents.
    We only maintain one set of NightEvents for a site at any given time.
    """
    _night_events: Dict[Site, NightEvents] = {}

    @staticmethod
    def get_night_events(time_grid: Time,
                         time_slot_length: TimeDelta,
                         site: Site) -> NightEvents:
        """
        Retrieve NightEvents. These may contain more information than requested,
        but never less.
        """
        ne = NightEventsManager._night_events

        # Recalculate if we are not compatible.
        if (site not in ne or
                time_slot_length != ne[site].time_slot_length or
                (len(ne[site].time_grid) == 1 and ne[site].time_grid[0] != time_grid[0]) or
                (len(ne[site].time_grid) > 1 and
                 (time_grid[0] < ne[site].time_grid[0] or time_grid[-1] > ne[site].time_grid[-1]))):
            # For some strange reason, this does not work if we specify keywords for NightEvents.
            # It complains about __init__() getting multiple args for time_grid.
            night_events = NightEvents(
                time_grid,
                time_slot_length,
                site,
                *sky.night_events(time_grid, site.location, site.timezone)
            )
            NightEventsManager._night_events[site] = night_events

        return NightEventsManager._night_events[site]
