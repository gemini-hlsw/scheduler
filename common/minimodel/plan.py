from common.calculations.nightevents import NightEvents
from common.minimodel.visit import Visit
from common.minimodel.site import Site
from common.minimodel.observation import Observation
from dataclasses import dataclass
from datetime import timedelta, datetime
from math import ceil
from typing import NoReturn, Mapping, List


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
    A collection of Plan from all sites for a specific night
    """
    def __init__(self, night_events: Mapping[Site, NightEvents], night_idx: int):

        self.plans = {}
        self.night = night_idx
        for site, ne in night_events.items():
            if ne is not None:
                self.plans[site] = Plan(ne.local_times[night_idx][0],
                                        ne.local_times[night_idx][-1],
                                        ne.time_slot_length,
                                        site,
                                        len(ne.times[night_idx]))
    
    def __getitem__(self, site: Site) -> Plan:
        return self.plans[site]

    def __iter__(self):
        return iter(self.plans.values())

    def all_done(self) -> bool:
        """
        Check if all plans for all sites are done in a night
        """
        return all(plan.is_full for plan in self.plans.values())
    
