# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
import sys
from typing import Dict, List

from lucupy.minimodel import TimeslotIndex, NightIndex, Site

from scheduler.core.eventsqueue import Event
from scheduler.core.plans import Plans, Plan


@dataclass
class NightChanges:
    lookup: Dict[Event, Plans] = field(init=False, default_factory=dict)

    def get_final_plans(self):
        return list(self.lookup.values())[-1]


@dataclass(frozen=True)
class TimelineEntry:
    start_time_slot: TimeslotIndex
    event: Event
    plan_generated: Plan


@dataclass
class NightlyTimeline:
    """
    A collection of timeline entries per night and site.
    """
    timeline: Dict[NightIndex, Dict[Site, List[TimelineEntry]]] = field(init=False, default_factory=dict)

    def add(self,
            night_idx: NightIndex,
            site: Site,
            time_slot: TimeslotIndex,
            event: Event,
            plan_generated: Plan) -> None:
        entry = TimelineEntry(time_slot,
                              event,
                              plan_generated)
        self.timeline.setdefault(night_idx, {}).setdefault(site, []).append(entry)

    def get_final_plan(self, night_idx: NightIndex, site: Site) -> Plan:
        if night_idx not in self.timeline:
            raise RuntimeError(f'Cannot get final plan: {night_idx} for site {site.name} not in timeline.')
        if site not in self.timeline[night_idx]:
            raise RuntimeError(f'Cannot get final plan: {site.name} not in timeline.')
        entries = self.timeline[night_idx][site]

        all_generated = []
        t = 0

        for entry in reversed(entries):
            pg = entry.plan_generated

            if t > 0:
                # grab partial plan at the timeslot
                partial_plan = pg.get_slice(stop=t)
                # modfy reflect last visit to match starting time_slot from the visit
                # TODO: this need to be reflected in Optimizer
                partial_plan.visits[-1].time_slots = t - partial_plan.visits[-1].start_time_slot
            else:
                partial_plan = pg

            all_generated += [v for v in reversed(partial_plan.visits)]
            if t < entry.start_time_slot:
                t = entry.start_time_slot

        p = Plan(start=entries[0].plan_generated.start,
                 end=entries[-1].plan_generated.end,
                 time_slot_length=entries[0].plan_generated.time_slot_length,
                 site=site,
                 _time_slots_left=entries[-1].plan_generated.time_left())
        p.visits = [v for v in reversed(all_generated)]
        return p

    def display(self) -> None:
        sys.stderr.flush()
        for night_idx, entries_by_site in self.timeline.items():
            print(f'\n\n+++++ NIGHT {night_idx + 1} +++++')
            for site, entries in sorted(entries_by_site.items(), key=lambda x: x[0].name):
                for entry in entries:
                    print(f'\t+++++ Triggered by event: {entry.event.reason} at {entry.start_time_slot} on {site} +++++')
                    # print(f'Plan for site: {plan.site.name}')
                    for visit in entry.plan_generated.visits:
                        print(
                            f'\t{visit.start_time}   {visit.obs_id.id:20} {visit.score:8.2f} {visit.atom_start_idx:4d} '
                            f'{visit.atom_end_idx:4d} {visit.start_time_slot:4d}')
                    print('\t+++++ END EVENT +++++')
        sys.stdout.flush()
