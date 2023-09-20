# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import List, Mapping, Optional, Tuple

from lucupy.minimodel import Band, Conditions, NightIndex, Observation, ObservationID, Resource, Site, ObservationClass
import numpy as np
import numpy.typing as npt

from scheduler.core.calculations import TargetInfoNightIndexMap
from scheduler.core.calculations.nightevents import NightEvents


@dataclass(order=True, frozen=True)
class Visit:
    start_time: datetime  # Unsure if this or something else
    obs_id: ObservationID
    obs_class: ObservationClass
    atom_start_idx: int
    atom_end_idx: int
    start_time_slot: int
    time_slots: int
    score: float
    instrument: Optional[Resource]


@dataclass(frozen=True)
class NightStats:
    time_loss: str
    plan_score: float
    plan_conditions: Conditions
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

    def __post_init__(self):
        self.visits: List[Visit] = []
        self.is_full = False
        self.night_stats: Optional[NightStats] = None
        self.alt_degs: List[List[float]] = []

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

            n_slots_acq = Plan.time2slots(self.time_slot_length, obs.acq_overhead)
            visit_length = n_slots_acq + Plan.time2slots(self.time_slot_length,
                                                         cumul_seq[atom_end])

            # TODO: can this be done w/o a loop? convert cumm_seq to slots, and find the value that fits
            while n_slots_filled + visit_length <= len_interval and atom_end <= len(cumul_seq) - 2:
                atom_end += 1
                visit_length = n_slots_acq + \
                    Plan.time2slots(self.time_slot_length, cumul_seq[atom_end])

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

    def _place_standards(self,
                         interval: npt.NDArray[int],
                         science_obs: List[Observation],
                         partner_obs: List[Observation],
                         target_info: TargetInfoNightIndexMap,
                         night_idx: NightIndex,
                         n_std) -> Tuple[List, List]:
        """Pick the standards that best match the NIR science
        observations by airmass
        """

        standards = []
        placement = []

        xdiff_min = xdiff_before_min = xdiff_after_min = 99999.
        std_before = None
        std_after = None
        # If only one standard needed, try before or after, use best airmass match
        # TODO: Any preference to taking the standard before or after?
        # TODO: Check scores to confirm that the observations are scheduleable (?)
        for partcal_obs in partner_obs:
            # Need the length of the calibration sequence only
            n_slots_cal = Plan.time2slots(self.time_slot_length, partcal_obs.exec_time())
            n_slots_acq = Plan.time2slots(self.time_slot_length, partcal_obs.acq_overhead)

            # Try std first
            # Mean std airmass
            slot_start = n_slots_acq
            slot_end = n_slots_cal - 1

            xmean_cal = target_info[night_idx][partcal_obs.id].mean_airmass(interval[slot_start:slot_end + 1])

            # Mean NIR science airmass
            idx_start_nir, idx_end_nir, obs_id_nir = self.nir_slots(science_obs, n_slots_cal, len(interval))
            slot_start_nir = slot_end + idx_start_nir
            slot_end_nir = slot_end + idx_end_nir

            xmean_nir = target_info[night_idx][obs_id_nir].mean_airmass(interval[slot_start_nir:slot_end_nir + 1])
            xdiff_before = np.abs(xmean_nir - xmean_cal)

            # Try std last
            # Mean std airmass
            len_int = len(interval)
            slot_start = len_int - 1 - n_slots_cal + n_slots_acq
            slot_end = slot_start + n_slots_cal - n_slots_acq - 1

            xmean_cal = (target_info[night_idx][partcal_obs.id][night_idx]
                         .mean_airmass(interval[slot_start:slot_end + 1]))

            # Mean NIR science airmass
            slot_start_nir = idx_start_nir
            slot_end_nir = idx_end_nir

            xmean_nir = target_info[night_idx][obs_id_nir].mean_airmass(interval[slot_start_nir:slot_end_nir + 1])
            xdiff_after = np.abs(xmean_nir - xmean_cal)

            if n_std == 1:
                if xdiff_before <= xdiff_after:
                    xdiff = xdiff_before
                    place_before = True  # before
                else:
                    xdiff = xdiff_after
                    place_before = False  # after

                if xdiff < xdiff_min:
                    xdiff_min = xdiff
                    placement = [place_before]
                    standards = [partcal_obs]
            else:
                if xdiff_before < xdiff_before_min:
                    xdiff_before_min = xdiff_before
                    std_before = partcal_obs

                if xdiff_after < xdiff_after_min:
                    xdiff_after_min = xdiff_after
                    std_after = partcal_obs
        # TODO: This should be added directly to the plan
        if n_std > 1:
            placement = [True, False]
            standards = [std_before, std_after]

        return standards, placement

    @staticmethod
    def time2slots(time_slot_length: timedelta, time: timedelta) -> int:
        # return ceil((time.total_seconds() / self.time_slot_length.total_seconds()) / 60)
        # return ceil((time.total_seconds() / self.time_slot_length.total_seconds()))
        return ceil(time / time_slot_length)

    def add(self,
            obs: Observation,
            start: datetime,
            atom_start: int,
            atom_end: int,
            start_time_slot: int,
            time_slots: int,
            score: float) -> None:
        visit = Visit(start,
                      obs.id,
                      obs.obs_class,
                      atom_start,
                      atom_end,
                      start_time_slot,
                      time_slots,
                      score,
                      obs.instrument())
        self.visits.append(visit)
        self._time_slots_left -= time_slots

    def time_left(self) -> int:
        return self._time_slots_left

    def update_time_slots(self, time: int):
        self._time_slots_left = time

    def __contains__(self, obs: Observation) -> bool:
        return any(visit.obs_id == obs.id for visit in self.visits)


class Plans:
    """
    A collection of Plan from all sites for a specific night
    """

    def __init__(self, night_events: Mapping[Site, NightEvents], night_idx: NightIndex):
        self.plans = {}
        self.night_idx = night_idx
        for site, ne in night_events.items():
            if ne is not None:
                self.plans[site] = Plan(ne.local_times[night_idx][0],
                                        ne.local_times[night_idx][-1],
                                        ne.time_slot_length.to_datetime(),
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
