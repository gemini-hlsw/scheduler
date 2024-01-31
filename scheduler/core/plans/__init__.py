# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field, InitVar
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Tuple

from lucupy.minimodel import Band, NightIndex, Observation, ObservationID, Resource, Site, ObservationClass
from lucupy.timeutils import time2slots

from scheduler.core.calculations.nightevents import NightEvents


@dataclass(order=True)
class Visit:
    start_time: datetime  # Unsure if this or something else
    obs_id: ObservationID
    obs_class: ObservationClass
    atom_start_idx: int
    atom_end_idx: int
    start_time_slot: int
    time_slots: int
    score: float
    peak_score: float
    instrument: Optional[Resource]
    completion: str


@dataclass(frozen=True)
class NightStats:
    time_loss: str
    plan_score: float
    n_toos: int
    completion_fraction: Mapping[Band, int]


@dataclass
class Plan:
    """
    Nightly plan for a specific Site.

    night_stats are supposed to be calculated at the end when plans are already generated.
    """
    start: datetime
    end: datetime
    time_slot_length: timedelta
    site: Site
    _time_slots_left: int

    visits: List[Visit] = field(init=False, default_factory=lambda: [])
    is_full: bool = field(init=False, default=False)
    night_stats: Optional[NightStats] = field(init=False, default=None)
    alt_degs: List[List[float]] = field(init=False, default_factory=lambda: [])

    def nir_slots(self,
                  science_obs: List[Observation],
                  n_slots_filled: int,
                  len_interval: int) -> Tuple[int, int, ObservationID]:
        """
        Return the starting and ending timeline slots (indices) for the NIR science observations
        """
        # TODO: This should probably be moved to a more general location
        nir_inst = [Resource('Flamingos2'), Resource('GNIRS'), Resource('NIRI'), Resource('NIFS'),
                    Resource('IGRINS')]

        # science, split at atom
        slot_start_nir = None
        slot_end_nir = None
        slot_start = 0
        obs_id_nir = None

        for obs in science_obs:
            obs_id = obs.id

            # TODO: currently in lucupy
            cumul_seq = []
            atom_start = 0
            atom_end = atom_start

            n_slots_acq = time2slots(self.time_slot_length, obs.acq_overhead)
            visit_length = n_slots_acq + time2slots(self.time_slot_length, cumul_seq[atom_end])

            # TODO: can this be done w/o a loop? convert cumm_seq to slots, and find the value that fits
            while n_slots_filled + visit_length <= len_interval and atom_end <= len(cumul_seq) - 2:
                atom_end += 1
                visit_length = n_slots_acq + time2slots(self.time_slot_length, cumul_seq[atom_end])

            slot_end = slot_start + visit_length - 1
            # NIR science time for to determine the number of tellurics
            if any(inst in obs.required_resources() for inst in nir_inst):
                if slot_start_nir is None:
                    slot_start_nir = slot_start + n_slots_acq  # start of the science sequence, after acq
                slot_end_nir = slot_end
                obs_id_nir = obs_id

            n_slots_filled += visit_length

            slot_start = slot_end + 1  # for the next iteration

        return slot_start_nir, slot_end_nir, obs_id_nir

    def add(self,
            obs: Observation,
            start: datetime,
            atom_start: int,
            atom_end: int,
            start_time_slot: int,
            time_slots: int,
            score: float,
            peak_score: float) -> None:
        visit = Visit(start,
                      obs.id,
                      obs.obs_class,
                      atom_start,
                      atom_end,
                      start_time_slot,
                      time_slots,
                      score,
                      peak_score,
                      obs.instrument(),
                      f'{atom_end+1}/{len(obs.sequence)}')
        self.visits.append(visit)
        self._time_slots_left -= time_slots

    def time_left(self) -> int:
        return self._time_slots_left

    def update_time_slots(self, time: int):
        self._time_slots_left = time

    def get_slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> 'Plan':
        if not start and not stop:
            return self
        else:
            visits_by_timeslot = {v.start_time_slot: v for v in self.visits}
            visits_timeslots = [v.start_time_slot for v in self.visits]
            plan = Plan(self.start, self.end, self.time_slot_length, self.site, self._time_slots_left)

            start = start or 0
            stop = stop or visits_timeslots[-1]
            plan.visits = [visits_by_timeslot[x] for x in visits_timeslots if
                           start <= x <= stop]
            return plan

    def __contains__(self, obs: Observation) -> bool:
        return any(visit.obs_id == obs.id for visit in self.visits)


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
