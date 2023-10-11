from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, List

import numpy as np
from lucupy.minimodel import TimeslotIndex, NightIndex
from lucupy.types import Interval

from scheduler.core.eventsqueue import Event
from scheduler.core.plans import Plans


@dataclass
class NightChanges:
    lookup: Dict[Event, Plans] = field(default_factory=dict)

    def get_final_plans(self):
        return list(self.lookup.values())[-1]


@dataclass(frozen=True)
class TimelineEntry:
    start_time_slots: TimeslotIndex
    event: Event
    plans_generated: Plans

@dataclass
class NightTimeline:
    timeline: Dict[NightIndex, List[TimelineEntry]]

    def add(self,
            night_idx: NightIndex,
            time_slot: TimeslotIndex,
            event: Event,
            plans_generated: Plans) -> None:

        entry = TimelineEntry(time_slot,
                              event,
                              plans_generated)
        if night_idx in self.timeline:
            self.timeline[night_idx].append(entry)
        else:
            self.timeline[night_idx] = [entry]

    def get_final_plans(self, night_idx) -> Plans:
        try:
            return self.timeline[night_idx][-1].plans_generated
        except KeyError:
            raise KeyError(f'Missing plans for night {night_idx}')

    def display(self) -> None:

        for night_idx, entries in self.timeline.items():
            print(f'\n\n+++++ NIGHT {night_idx + 1} +++++')
            for entry in entries:
                print(f'\t+++++ Triggered by event: {entry.event.reason} at {entry.event.start} +++++')
                for plan in entry.plans_generated:
                    print(f'Plan for site: {plan.site.name}')
                    for visit in plan.visits:
                        print(
                            f'\t{visit.start_time}   {visit.obs_id.id:20} {visit.score:8.2f} {visit.atom_start_idx:4d} '
                            f'{visit.atom_end_idx:4d}')
                print('\t+++++ END EVENT +++++')

