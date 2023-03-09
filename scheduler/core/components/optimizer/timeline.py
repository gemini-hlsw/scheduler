EMPTY = -1
UNSCHEDULABLE = -2

from dataclasses import dataclass
from typing import List, NoReturn, Mapping, Optional, Sized, Tuple
from datetime import datetime, timedelta

from lucupy.minimodel import Observation, ObservationID, Site
from scheduler.core.calculations.nightevents import NightEvents

import numpy as np


@dataclass
class Timeline:
    """
    Nightly plan for a specific Site for the GreedyMax optimizer. Each plan is a timeline array with one
    entry for each time slot for the night. Each value needs to be a pointer or index to the observation
    scheduled in that slot.
    """
    start: datetime
    end: datetime
    time_slot_length: timedelta
    site: Site
    _total_time_slots: int

    def __post_init__(self):
        self.schedule = np.full(self._total_time_slots, EMPTY)
        self.is_full = False

    def __contains__(self, obs: Observation) -> bool:
        return any(visit.obs_id == obs.id for visit in self.visits)


class Timelines:
    """
    A collection of Timeline from all sites for a specific night
    """

    def __init__(self, night_events: Mapping[Site, NightEvents], night_idx: int):

        self.timelines = {}
        self.night = night_idx
        for site, ne in night_events.items():
            if ne is not None:
                self.timelines[site] = Timeline(ne.local_times[night_idx][0],
                                        ne.local_times[night_idx][-1],
                                        ne.time_slot_length.to_datetime(),
                                        site,
                                        len(ne.times[night_idx]))

    def __getitem__(self, site: Site) -> Timeline:
        return self.timelines[site]

    def __iter__(self):
        return iter(self.timelines.values())

    def all_done(self) -> bool:
        """
        Check if all plans for all sites are done in a night
        """
        return all(timeline.is_full for timeline in self.timelines.values())