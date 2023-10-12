from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, List

import numpy as np
from lucupy.minimodel import TimeslotIndex, NightIndex, Site
from lucupy.types import Interval

from scheduler.core.eventsqueue import Event
from scheduler.core.plans import Plans, Plan


@dataclass
class NightChanges:
    lookup: Dict[Event, Plans] = field(default_factory=dict)

    def get_final_plans(self):
        return list(self.lookup.values())[-1]


@dataclass(frozen=True)
class TimelineEntry:
    start_time_slots: TimeslotIndex
    event: Event
    plan_generated: Plan

@dataclass
class NightTimeline:
    timeline: Dict[NightIndex, Dict[Site, List[TimelineEntry]]]

    def add(self,
            night_idx: NightIndex,
            site: Site,
            time_slot: TimeslotIndex,
            event: Event,
            plan_generated: Plan) -> None:

        entry = TimelineEntry(time_slot,
                              event,
                              plan_generated)
        if night_idx in self.timeline:
            self.timeline[night_idx][site].append(entry)
        else:
            self.timeline[night_idx] = {site: [entry]}

    def get_final_plan(self, night_idx, site) -> Plans:

        entries = self.timeline[night_idx][site]
        for entry in entries:
            plan_generated = entry.start_time_slots

        try:

            for e in self.timeline:


            return self.timeline[night_idx]
        except KeyError:
            raise KeyError(f'Missing plans for night {night_idx}')

    def display(self) -> None:

        for night_idx, entries_by_site in self.timeline.items():
            print(f'\n\n+++++ NIGHT {night_idx + 1} +++++')
            for site, entries in entries_by_site.items():
                for entry in entries:
                    print(f'\t+++++ Triggered by event: {entry.event.reason} at {entry.event.start} on {site} +++++')
                    for plan in entry.plan_generated:
                        print(f'Plan for site: {plan.site.name}')
                        for visit in entry.plan_generated.visits:
                            print(
                                f'\t{visit.start_time}   {visit.obs_id.id:20} {visit.score:8.2f} {visit.atom_start_idx:4d} '
                                f'{visit.atom_end_idx:4d}')
                    print('\t+++++ END EVENT +++++')

