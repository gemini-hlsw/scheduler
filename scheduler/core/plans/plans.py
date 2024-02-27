# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field, InitVar
from typing import final, Dict, Mapping

from lucupy.minimodel import NightIndex, Site

from scheduler.core.calculations.nightevents import NightEvents


__all__ = [
    'Plans',
]


@final
@dataclass
class Plans:
    """
    A collection of Plan for all sites for a specific night.
    """
    night_events: InitVar[Mapping[Site, NightEvents]]
    night_idx: NightIndex
    plans: Dict[Site, Plan] = field(init=False, default_factory=lambda: {})

    def __post_init__(self, night_events: Mapping[Site, NightEvents]):
        self.plans: Dict[Site, Plan] = {}
        for site, ne in night_events.items():
            if ne is not None:
                self.plans[site] = Plan(ne.local_times[self.night_idx][0],
                                        ne.local_times[self.night_idx][-1],
                                        ne.time_slot_length.to_datetime(),
                                        site,
                                        len(ne.times[self.night_idx]))

    def __getitem__(self, site: Site) -> Plan:
        return self.plans[site]

    def __setitem__(self, key: Site, value: Plan) -> None:
        self.plans[key] = value

    def __iter__(self):
        return iter(self.plans.values())

    def all_done(self) -> bool:
        """
        Check if all plans for all sites are done in a night
        """
        return all(plan.is_full for plan in self.plans.values())
