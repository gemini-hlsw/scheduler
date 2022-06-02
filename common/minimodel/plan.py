from dataclasses import dataclass
from datetime import timedelta, datetime

from common.minimodel.visit import Visit
from common.minimodel.site import Site
from common.minimodel.observation import Observation
from math import ceil
from typing import NoReturn



@dataclass
class Plan:
    """
    Nightly plan for a specific Site
    """
    start: datetime
    end: datetime
    time_slot_length: timedelta
    site: Site
    _time_slots_left: int
    
    def __post_init__(self):
        self.visits = []
        self.is_full = False
    
    def time2slots(self, time: datetime) -> int:
        return ceil((time.total_seconds() / 60) / self._time_slot_length.value)

    def add(self, obs: Observation, start: datetime, time_slots: int) -> NoReturn:
        visit = Visit(start, obs.id, obs.sequence[0].id, obs.sequence[-1].id)
        self.visits.append(visit)
        self._time_slots_left -= time_slots

    def time_left(self) -> int:
        return self._time_slots_left


class Plans:
    """
    A collection of Nightly Plan from all sites
    """
    def __init__(self, night_events):
        # TODO: adding NightEvents creates a circular dependency!

        # TODO: Assumes that all sites schedule the same amount of nights
        self.nights = [[] for _ in range(len(night_events.values()[0].time_grid))]

        for site in night_events.keys():
            if night_events[site] is not None:
                for idx, jdx in night_events[site].time_grid:
                    self.nights[idx].append(Plan(night_events[site].local_times[idx][0],
                                                 night_events[site].local_times[idx][-1],
                                                 night_events[site].time_slot_length[idx],
                                                 site))
    
    def all_done(self, night: int) -> bool:
        """
        Check if all plans for all sites are done in that night
        """
        return all(plan.is_full for plan in self.nights[night])
