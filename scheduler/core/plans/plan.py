# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import final, List, Optional, Tuple

from lucupy.minimodel import Observation, ObservationID, Resource, Site, Variant
from lucupy.timeutils import time2slots

from .nightstats import NightStats
from .visit import Visit


__all__ = [
    'Plan',
]


@final
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

    visits: List[Visit] = field(init=False, default_factory=list)
    is_full: bool = field(init=False, default=False)
    night_stats: Optional[NightStats] = field(init=False, default=None)
    alt_degs: List[List[float]] = field(init=False, default_factory=list)

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
            peak_score: float,
            current_conditions: Variant) -> None:
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
                      f'{atom_end+1}/{len(obs.sequence)}',
                      current_conditions)
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
