# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import final, ClassVar, Dict, List, Optional
from zoneinfo import ZoneInfo

from lucupy.minimodel import TimeslotIndex, NightIndex, Site
from pandas.io.stata import excessive_string_length_error

from scheduler.core.events.queue import Event, InterruptionResolutionEvent, FaultResolutionEvent, \
    WeatherClosureResolutionEvent, MorningTwilightEvent
from scheduler.core.plans import Plan


__all__ = [
    'TimelineEntry',
    'NightlyTimeline'
]


@final
@dataclass(frozen=True)
class TimelineEntry:
    start_time_slot: TimeslotIndex
    event: Event
    plan_generated: Optional[Plan]


@final
@dataclass
class NightlyTimeline:
    """
    A collection of timeline entries per night and site.
    """
    timeline: Dict[NightIndex, Dict[Site, List[TimelineEntry]]] = field(init=False, default_factory=dict)
    time_losses: Dict[NightIndex, Dict[Site, Dict[str, int]]] = field(init=False, default_factory=dict)
    _datetime_formatter: ClassVar[str] = field(init=False, default='%Y-%m-%d %H:%M')

    def add(self,
            night_idx: NightIndex,
            site: Site,
            time_slot: TimeslotIndex,
            event: Event,
            plan_generated: Optional[Plan]) -> None:
        entry = TimelineEntry(time_slot,
                              event,
                              plan_generated)
        self.timeline.setdefault(night_idx, {}).setdefault(site, []).append(entry)

    def get_final_plan(self,
                       night_idx: NightIndex,
                       site: Site,
                       is_unblocked: bool) -> Optional[Plan]:
        """Get the final plan after all the events are processed and the scheduler
            reaches the Morning Twilight.

            Args:
                night_idx (NightIndex): Night index that the plan belongs to.
                site (Site): The site that the plan belongs to.
                is_unblocked (bool): Whether the site is unblocked or not.
        """
        if night_idx not in self.timeline:
            raise RuntimeError(f'Cannot get final plan: {night_idx} for site {site.name} not in timeline.')
        if site not in self.timeline[night_idx]:
            raise RuntimeError(f'Cannot get final plan: {site.name} not in timeline.')
        entries = self.timeline[night_idx][site]

        all_generated = []

        # Skip the None entries.
        relevant_entries = [e for e in reversed(entries) if e.plan_generated is not None]
        if len(relevant_entries) == 0:
            return None

        t = relevant_entries[0].start_time_slot
        for i, entry in enumerate(relevant_entries):
            pg = entry.plan_generated
            if i > 0:
                # Get the partial plan, i.e. all visits up to and including time slot t.
                # print("start timeslot: ",t)
                partial_plan = pg.get_slice(stop=t)
                # print("partial plan: ",partial_plan.visits)
                # If there was a last_visit (i.e. partial_plan.visits[-1]), then it:
                # * started at last_visit.start_time_slot;
                # * was scheduled for last_visit.time_slots; and thus
                # * ended at last_visit.start_time_slot + last_visit.time_slots.
                # If the visit was cut off, i.e:
                #     t in [last_visit.start_time_slot, last_end_time_slot)
                # then mark the actual number of time slots the visit was able to run.
                # TODO: this need to be reflected in Optimizer
                if partial_plan.visits:
                    last_visit = partial_plan.visits[-1]
                    last_start_time_slot = last_visit.start_time_slot
                    last_end_time_slot = last_start_time_slot + last_visit.time_slots
                    if last_start_time_slot <= t < last_end_time_slot:
                        last_visit.time_slots = t - last_start_time_slot
                t = entry.start_time_slot
            else:
                if is_unblocked:
                    partial_plan = pg
                else:
                    partial_plan = deepcopy(pg)
                    partial_plan.visits = []
            all_generated += [v for v in reversed(partial_plan.visits) if v.time_slots]

        # Evening twilight plan
        eve_plan = relevant_entries[0].plan_generated

        night_time_slots = (eve_plan.end - eve_plan.start) // eve_plan.time_slot_length
        used_time_slots = sum([v.time_slots for v in reversed(all_generated)])

        p = Plan(start=relevant_entries[0].plan_generated.start,
                 end=relevant_entries[-1].plan_generated.end,
                 time_slot_length=relevant_entries[0].plan_generated.time_slot_length,
                 site=site,
                 _time_slots_left=night_time_slots - used_time_slots,
                 conditions=relevant_entries[-1].plan_generated.conditions)
        p.visits = [v for v in reversed(all_generated)]
        return p

    def calculate_time_losses(self, night_idx: NightIndex, site: Site) -> None:
        """Calculates the time lost by different types of events for the night.

          Args:
          night_idx (NightIdx): Night index of the plan.
          site (Site): Site for the night plan.

          Returns:
            None: Updates the `time_losses` attribute.

        """
        # Initialize the values
        self.time_losses.setdefault(night_idx, {})
        self.time_losses[night_idx].setdefault(site, {})
        self.time_losses[night_idx][site].setdefault("weather", 0)
        self.time_losses[night_idx][site].setdefault("fault", 0)
        self.time_losses[night_idx][site].setdefault("unschedule", 0)

        weather = 0
        fault = 0
        for entry in self.timeline[night_idx][site]:
            event = entry.event
            if isinstance(event, InterruptionResolutionEvent):
                match event:
                    case FaultResolutionEvent():
                        fault += event.time_loss.total_seconds() // 60
                    case WeatherClosureResolutionEvent():
                        # TODO: Weather is giving float somehow
                        weather += int(event.time_loss.total_seconds() // 60)

        # This is to ensure no matter what order the ResolutionEvents are we get all at them accounted.
        for entry in self.timeline[night_idx][site]:
            event = entry.event
            if isinstance(event, MorningTwilightEvent):
                if entry.plan_generated is not None:
                    unschedule = entry.plan_generated.time_left() - weather - fault
                    #  if unschedule < 0:
                    #    raise ValueError(f'Unscheduled time is negative!')
                    self.time_losses[night_idx][site]["unschedule"] = unschedule

        self.time_losses[night_idx][site]["weather"] = weather
        self.time_losses[night_idx][site]["fault"] = fault

    def display(self, output='stdout') -> None:
        def rnd_min(dt: datetime) -> datetime:
            return dt + timedelta(minutes=1 - (dt.minute % 1))

        sys.stderr.flush()
        f = sys.stdout if output == 'stdout' else open(output, 'w')
        for night_idx, entries_by_site in self.timeline.items():
            for site, entries in sorted(entries_by_site.items(), key=lambda x: x[0].name):
                print(f'\n\n+++++ NIGHT {night_idx + 1}, SITE: {site.name} +++++', file=f)
                for entry in entries:
                    time = rnd_min(entry.event.time).strftime(self._datetime_formatter)
                    print(f'\t+++++ Triggered by event: {entry.event.description} at {time} '
                          f'(time slot {entry.start_time_slot}) at site {site.name}', file=f)
                    if entry.plan_generated is not None:
                        for visit in entry.plan_generated.visits:
                            visit_time = rnd_min(visit.start_time).strftime(self._datetime_formatter)
                            print(f'\t{visit_time}   {visit.obs_id.id:20} {visit.score:8.2f} '
                                  f'{visit.atom_start_idx:4d} {visit.atom_end_idx:4d} {visit.start_time_slot:4d}'
                                  f' {visit.start_time_slot+visit.time_slots:4d}', file=f)
                    print('\t+++++ END EVENT +++++', file=f)
            print('', file=f)
        if f != sys.stdout:
            f.close()
        else:
            sys.stdout.flush()

    def to_json(self) -> dict:
        utc = ZoneInfo('UTC')
        return {
            n_idx: {site.name: [{'startTimeSlot': te.start_time_slot,
                                 'event': {'site': te.event.site.name,
                                           'time': te.event.time.strftime(self._datetime_formatter),
                                           'description': te.event.description,
                                           },
                                 'plan': {'start': te.plan_generated.start.astimezone(utc).strftime(self._datetime_formatter),
                                           'end': te.plan_generated.end.astimezone(utc).strftime(self._datetime_formatter),
                                           'site': te.plan_generated.site.name,
                                           'visits': [{"starTime": v.start_time.astimezone(utc).strftime(self._datetime_formatter),
                                                       "endTime": (v.start_time+
                                                                   v.time_slots*te.plan_generated.time_slot_length).strftime(self._datetime_formatter),
                                                       "obsId": v.obs_id.id,
                                                       "atomStartIdx": v.atom_start_idx,
                                                       "atomEndIdx": v.atom_end_idx,
                                                       "altitude": alt,
                                                       "instrument": v.instrument.id if v.instrument else '',
                                                       "obs_class": v.obs_class.name,
                                                       "score": v.score,
                                                       "peakScore": v.peak_score,
                                                       "completion": v.completion}
                                                      for v, alt in zip(te.plan_generated.visits, te.plan_generated.alt_degs)],
                                           'nightStats': {
                                               'timeLoss': te.plan_generated.night_stats.time_loss,
                                               'planScore': te.plan_generated.night_stats.plan_score,
                                               'completionFraction': te.plan_generated.night_stats.completion_fraction,
                                               'programCompletion': te.plan_generated.night_stats.program_completion
                                           }
                                          } if te.plan_generated else {}
                                 } for te in time_entries]
             for site, time_entries in by_site.items()
                    } for n_idx, by_site in self.timeline.items()
        }
