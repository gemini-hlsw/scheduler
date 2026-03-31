# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import final, Dict, FrozenSet, List, Optional, Tuple
from astropy.time import Time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from lucupy.minimodel import (Group, NightIndex, Observation, ObservationClass, ObservationID, ObservationStatus,
                              Program, QAState, Site, UniqueGroupID, Wavelengths, ObservationMode,
                              unique_group_id, AndOption, GROUP_NONE_ID)

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.timeutils import time2slots
from lucupy.types import Interval, ListOrNDArray, ZeroTime

from scheduler.core.calculations import GroupData, NightTimeslotScores, ProgramInfo
from scheduler.core.calculations.selection import Selection
from scheduler.core.components.optimizer.timeline import Timelines
from scheduler.core.plans import Plans
from scheduler.services import logger_factory
from .base import BaseOptimizer, MaxGroup


__all__ = [
    'GreedyMaxOptimizer',
]


logger = logger_factory.create_logger(__name__)


@final
@dataclass(frozen=True)
class ObsPlanData:
    """
    Storage to conglomerate observation information for the plan together.
    """
    obs: Observation
    obs_start: datetime
    obs_len: int
    atom_start: int
    atom_end: int
    visit_score: float
    peak_score: float


@final
class GreedyMaxOptimizer(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits for the rest of the night in a greedy fashion.
    """

    def __init__(self,
                 min_visit_len: timedelta = timedelta(minutes=30),
                 show_plots: bool = False, verbose: bool = False):
        self.selection: Optional[Selection] = None
        self.group_data_list: List[GroupData] = []
        self.group_ids: List[UniqueGroupID] = []
        self.obs_group_ids: List[UniqueGroupID] = []
        self.timelines: Dict[NightIndex, Timelines] = {}
        self.sites: FrozenSet[Site] = frozenset()
        self.obs_in_plan: Dict = {}
        self.min_visit_len = min_visit_len
        self.show_plots = show_plots
        self.verbose = verbose
        self.time_slot_length: Optional[timedelta] = None

    def setup(self, selection: Selection) -> GreedyMaxOptimizer:
        """
        Preparation for the optimizer.
        """
        self.selection = selection
        self.group_ids = list(selection.schedulable_groups)
        self.group_data_list = list(selection.schedulable_groups.values())
        # self._process_group_data(self.group_data_list)
        self.obs_group_ids = list(selection.obs_group_ids) # noqa
        self.timelines = {night_idx: Timelines(selection.night_events, night_idx)
                          for night_idx in selection.night_indices}
        self.sites = selection.sites
        self.time_slot_length = selection.time_slot_length
        for site in self.sites:
            self.obs_in_plan[site] = {}
        return self

    @staticmethod
    def non_zero_intervals(scores: NightTimeslotScores) -> npt.NDArray[int]:
        """
        Calculate the non-zero intervals in the data.
        This consists of an array with entries of the form [a, b] where
        the non-zero interval runs from a (inclusive) to b (exclusive).
        See test_greedymax.py for an example.

        The array returned here contains multiple Intervals and thus we leave the return type
        instead of using Interval.
        """
        # Create an array that is 1 where the score is greater than 0, and pad each end with an extra 0.
        not_zero = np.concatenate((np.array([0]), np.greater(scores, 0), np.array([0])))
        abs_diff = np.abs(np.diff(not_zero))

        # Return the ranges for each nonzero interval.
        return np.where(abs_diff == 1)[0].reshape(-1, 2)

    @staticmethod
    def _first_nonzero_time_idx(inlist: ListOrNDArray[timedelta]) -> int:
        """
        Find the index of the first nonzero timedelta in inlist
        Designed to work with the output from cumulative_seq_exec_times
        """
        for idx, value in enumerate(inlist):
            if value > ZeroTime:
                return idx
        return len(inlist) - 1

    @staticmethod
    def num_nir_standards(exec_sci: timedelta,
                          wavelengths: Wavelengths = frozenset(),
                          mode: ObservationMode = ObservationMode.LONGSLIT) -> int:
        """
        Calculated the number of NIR standards from the length of the NIR science and the mode
        """
        n_std = 0

        # TODO: need mode or other info to distinguish imaging from spectroscopy
        if mode == ObservationMode.IMAGING:
            time_per_standard = timedelta(hours=2.0)
        else:
            if all(wave <= 2.5 for wave in wavelengths):
                time_per_standard = timedelta(hours=1.5)
            else:
                time_per_standard = timedelta(hours=1.0)

        if time_per_standard > ZeroTime:
            n_std = max(1, int(exec_sci // time_per_standard))  # TODO: confirm this

        return n_std

    def _exec_time_remaining(self,
                             group: Group,
                             verbose: bool = False) -> Tuple[timedelta, timedelta, timedelta, int, int]:
        """Determine the total and minimum remaining execution times.
           If an observation can't be split, then there should only be one atom, so min time is the full time.
           """

        if verbose:
            print('_exec_time_remaining')
            print(f"Group {group.unique_id.id} {group.exec_time()} {group.is_observation_group()} "
                  f"{group.is_scheduling_group()} {group.number_to_observe}")
            print(f"\t {group.required_resources()}")
            print(f"\t {group.wavelengths()}")

        nsci = nprt = 0

        exec_sci_min = exec_sci_nir = ZeroTime
        exec_prt = ZeroTime
        time_per_standard = ZeroTime
        sci_times = ZeroTime
        n_std = 0
        n_slots_remaining = 0
        part_times = []
        sci_times_min = []

        for obs in group.observations():
            # Unobserved remaining time, cumulative sequence of atoms
            cumul_seq = obs.cumulative_exec_times()
            if verbose:
                print(f"\t Obs: {obs.id.id} {obs.exec_time()} {obs.obs_class.name} {obs.site.name} "
                      f"{next(iter(obs.wavelengths()))} {cumul_seq[-1]}")
            #               f"{next(iter(obs.required_resources())).id} {next(iter(obs.wavelengths()))}")

            if cumul_seq[-1] > ZeroTime:
                # total time remaining
                time_remain = obs.acq_overhead + cumul_seq[-1]
                # Min time remaining (acq + first non-zero atom)
                time_remain_min = obs.acq_overhead + cumul_seq[self._first_nonzero_time_idx(cumul_seq)]

                if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
                    # Calculate the program time remaining, we won't split program standards
                    nsci += 1
                    sci_times += time_remain
                    # Add number of slots, to avoid rounding problems
                    n_slots_remaining += time2slots(self.time_slot_length, time_remain)
                    if obs.obs_class == ObservationClass.SCIENCE:
                        sci_times_min.append(time_remain_min)
                    else:
                        sci_times_min.append(time_remain)

                    # NIR science time for to determine the number of tellurics
                    if any(inst in obs.required_resources() for inst in ObservatoryProperties.nir_instruments()):
                            # and obs.obs_mode() in [ObservationMode.LONGSLIT, ObservationMode.XD, ObservationMode.MOS]:
                        exec_sci_nir += time_remain
                        if verbose:
                            print(f'Adding {time_remain} to exec_sci_nir')
                elif obs.obs_class == ObservationClass.PARTNERCAL:
                    # Partner calibration time, no splitting of partner cals
                    nprt += 1
                    part_times.append(time_remain)

        # Times for science observations
        exec_sci = sci_times
        if (nsci > 1) or (nsci > 0 and nprt > 0):
            # Don't split if more than one science/program standard obs, assume consecutive AND group
            # We also won't split NIR observations w/stds for now
            # TODO: support NIR obs splitting and then dynamically find the number of standards needed
            exec_sci_min = exec_sci
        elif nsci == 1:
            exec_sci_min = sci_times_min[0]

        # How many standards are needed?
        # TODO: need mode or other info to distinguish imaging from spectroscopy
        if exec_sci_nir > ZeroTime and len(part_times) > 0:
            n_std = self.num_nir_standards(exec_sci_nir, wavelengths=group.wavelengths(), mode=group.obs_mode())

        # if only partner standards, set n_std to the number of standards in group (e.g. specphots)
        # ToDo: review this... can cause problems depending on use of n_std
        if nprt > 0 and nsci == 0:
            n_std = nprt

        # most conservative, at the moment we don't know which standards will be picked,
        # the same star could be picked for before and after
        if n_std >= 1:
            exec_prt = n_std * max(part_times)
            # Slots needed, avoid rounding problems
            n_slots_remaining += n_std * time2slots(self.time_slot_length, max(part_times))

        exec_remain = exec_sci + exec_prt
        exec_remain_min = exec_sci_min + exec_prt

        if verbose:
            print(f"\t nsci = {nsci} {exec_sci} {exec_prt} nprt = {nprt} time_per_std = {time_per_standard}"
                  f" n_std = {n_std} n_slots_remaining = {n_slots_remaining}")
        #     print(f"\t n_std = {n_std} exec_remain = {exec_remain} exec_remain_min = {exec_remain_min}")

        return exec_remain, exec_remain_min, exec_sci_nir, n_std, n_slots_remaining

    def _min_slots_remaining(self, group: Group) -> Tuple[int, int, int, timedelta]:
        """
        Returns the minimum number of time slots for the remaining time.
        """

        # the number of time slots in the minimum visit length
        n_slots_min_visit = time2slots(self.time_slot_length, self.min_visit_len)
        # print(f"n_min_visit: {n_min_visit}")

        # Calculate the remaining clock time necessary for the group to be completed.
        # time_remaining = group.exec_time() - group.total_used()
        # the following does this
        time_remaining, time_remaining_min, exec_sci_nir, n_std, n_slots_remaining = self._exec_time_remaining(group)

        # Calculate the number of time slots needed to complete the group.
        # This use of time2slots works but is probably not kosher, need to make this more general.
        # n_slots_remaining = time2slots(self.time_slot_length, time_remaining)
        n_slots_remaining_min = time2slots(self.time_slot_length, time_remaining_min)

        # Time slots corresponding to the max of minimum time remaining and the minimum visit length
        # This supports long observations than cannot be split. helps prevent splitting into very small pieces
        n_min = max(n_slots_remaining_min, n_slots_min_visit)
        # Short groups should be done entirely, and we don't want to leave small pieces remaining
        # update the min useful time
        if n_slots_remaining - n_slots_min_visit <= n_slots_min_visit:
            n_min = n_slots_remaining

        return n_min, n_slots_remaining, n_std, exec_sci_nir

    def _find_max_group(self, plans: Plans) -> Optional[MaxGroup]:
        """
        Find the group with the max score in an open interval
        Returns None if there is no such group.
        Otherwise, returns a MaxGroup class containing information on the selected group.
        """

        # If true just analyze the only first open interval, like original GM, eventually make a parameter or setting
        only_first_interval = False

        # Get the unscheduled, available intervals (time slots)
        open_intervals = {site: self.timelines[plans.night_idx][site].get_available_intervals(only_first_interval)
                          for site in self.sites}

        max_scores = []
        groups = []
        intervals = []
        n_slots_remaining = []
        n_min_list = []
        n_std_list = []
        exec_nir_list = []

        # Make a list of scores in the remaining groups
        for group_data in self.group_data_list:
            site = group_data.group.observations()[0].site
            if not self.timelines[plans.night_idx][site].is_full and group_data.group.active:
                for interval_idx, interval in enumerate(open_intervals[site]):
                    # print(f'Interval: {iint}')
                    # scores = group_data.group_info.scores[plans.night]

                    # Get the maximum score over the interval.
                    smax = np.max(group_data.group_info.scores[plans.night_idx][interval])
                    if smax > 0.0:
                        # Check if the interval is long enough to be useful (longer than min visit length).
                        # Remaining time for the group.
                        # Also, should see if it can be split.
                        n_min, num_time_slots_remaining, n_std, exec_time_nir = \
                            self._min_slots_remaining(group_data.group)

                        # Evaluate sub-intervals (e.g. timing windows, gaps in the score).
                        # Find time slot locations where the score > 0.
                        # interval is a numpy array that indexes into the scores for the night to return a sub-array.
                        check_interval = group_data.group_info.scores[plans.night_idx][interval]
                        group_intervals = self.non_zero_intervals(check_interval)

                        max_score_on_interval = 0.0
                        max_interval = None
                        for group_interval in group_intervals:
                            grp_interval_length = group_interval[1] - group_interval[0]

                            max_score = np.max(group_data.group_info.scores[plans.night_idx]
                                               [interval[group_interval[0]:group_interval[1]]])

                            # Add a penalty if the length of the group is slightly more than the interval
                            # (within n_slots_min_visit), to discourage leaving small pieces behind
                            n_slots_min_visit = time2slots(self.time_slot_length, self.min_visit_len)
                            if 0 < num_time_slots_remaining - grp_interval_length < n_slots_min_visit:
                                max_score *= 0.5

                            # Find the max_score in the group intervals with non-zero scores
                            # The length of the non-zero interval must be at least as large as
                            # the minimum length
                            if max_score > max_score_on_interval and grp_interval_length >= n_min:
                                max_score_on_interval = max_score
                                max_interval = group_interval

                        if max_interval is not None:
                            max_scores.append(max_score_on_interval)
                            groups.append(group_data)
                            intervals.append(interval[max_interval[0]:max_interval[1]])
                            n_slots_remaining.append(num_time_slots_remaining)
                            n_min_list.append(n_min)
                            n_std_list.append(n_std)
                            exec_nir_list.append(exec_time_nir)

        max_score: Optional[float] = None
        max_group: Optional[GroupData] = None
        max_interval: Optional[Interval] = None
        max_n_min = None
        max_slots_remaining = None
        max_n_std = None
        max_exec_nir = ZeroTime

        if len(max_scores) > 0:
            # sort scores from high to low
            iscore_sort = np.flip(np.argsort(max_scores))
            ii = 0

            max_score = max_scores[iscore_sort[ii]]  # maximum score in time interval
            # consider groups with max scores within frac_score_limit of max_score
            # Only highest score: frac_score_limit = 0.0
            # Top 10%: frac_score_limit = 0.1
            # Consider everything: frac_score_limit = 1.0
            frac_score_limit = 0.1
            score_limit = max_score * (1.0 - frac_score_limit)
            max_group = groups[iscore_sort[ii]]
            max_interval = intervals[iscore_sort[ii]]
            max_n_min = n_min_list[iscore_sort[ii]]
            max_slots_remaining = n_slots_remaining[iscore_sort[ii]]
            max_n_std = n_std_list[iscore_sort[ii]]
            max_exec_nir = exec_nir_list[iscore_sort[ii]]

            # Prefer a group in the allowed score range if it does not require splitting,
            # otherwise take the top scorer
            selected = False
            while not selected and ii < len(iscore_sort):
                if (max_scores[iscore_sort[ii]] >= score_limit and
                        n_slots_remaining[iscore_sort[ii]] <= len(intervals[iscore_sort[ii]])):
                    # imax = ii
                    max_score = max_scores[iscore_sort[ii]]
                    max_group = groups[iscore_sort[ii]]
                    max_interval = intervals[iscore_sort[ii]]
                    max_n_min = n_min_list[iscore_sort[ii]]
                    max_slots_remaining = n_slots_remaining[iscore_sort[ii]]
                    max_n_std = n_std_list[iscore_sort[ii]]
                    max_exec_nir = exec_nir_list[iscore_sort[ii]]
                    selected = True
                ii += 1

        if max_score is None or max_group is None or max_interval is None:
            return None

        max_group_info = MaxGroup(
            group_data=max_group,
            max_score=max_score,
            interval=max_interval,
            n_min=max_n_min,
            n_slots_remaining=max_slots_remaining,
            n_std=max_n_std,
            exec_sci_nir=max_exec_nir,
            start_time=None,
            end_time=None
        )

        return max_group_info

    @staticmethod
    def _integrate_score(night_idx: NightIndex,
                         max_group_info: MaxGroup) -> Interval:
        """
        Use the score array to find the best location in the timeline
        """
        start = max_group_info.interval[0]
        end = max_group_info.interval[-1]
        scores = max_group_info.group_data.group_info.scores[night_idx]
        max_integral_score = scores[0]

        if len(max_group_info.interval) > 1:
            # Slide across the interval, integrating the score over the group length
            for idx in range(max_group_info.interval[0],
                             max_group_info.interval[-1] - max_group_info.n_slots_remaining + 2):

                integral_score = sum(scores[idx:idx + max_group_info.n_slots_remaining + 1])

                if integral_score > max_integral_score:
                    max_integral_score = integral_score
                    start = idx
                    end = start + max_group_info.n_slots_remaining - 1

        # Shift to window boundary if within minimum block time of edge.
        # If near both boundaries, choose boundary with higher score.
        score_start = scores[max_group_info.interval[0]]  # score at start of interval
        score_end = scores[max_group_info.interval[-1]]  # score at end of interval
        delta_start = start - max_group_info.interval[0]  # difference between start of window and block
        delta_end = max_group_info.interval[-1] - end  # difference between end of window and block
        # print(max_group_info.group_data.group.unique_id.id, score_start, score_end, delta_start, delta_end)

        # shift = None
        if delta_start < max_group_info.n_min and delta_end < max_group_info.n_min:
            if score_start > score_end and score_start > 0.0:
                start = max_group_info.interval[0]
                end = start + max_group_info.n_slots_remaining - 1
                # shift = 'left'
            elif score_end > 0.0:
                start = max_group_info.interval[-1] - max_group_info.n_slots_remaining + 1
                end = max_group_info.interval[-1]
                # shift = 'right'
        elif delta_start < max_group_info.n_min and score_start > 0.0:
            start = max_group_info.interval[0]
            end = start + max_group_info.n_slots_remaining - 1
            # shift = 'left'
        elif delta_end < max_group_info.n_min and score_end > 0:
            start = max_group_info.interval[-1] - max_group_info.n_slots_remaining + 1
            end = max_group_info.interval[-1]
            # shift = 'right'
        # if shift is not None:
        #     print(f'{max_group_info.group_data.group.unique_id.id} shifted {shift}, n_min = {max_group_info.n_min}')

        # Make final list of indices for the highest scoring shifted sub-interval
        best_interval = np.arange(start=start, stop=end+1)

        return best_interval

    @staticmethod
    def _find_group_position(night_idx: NightIndex, max_group_info: MaxGroup) -> Interval:
        """Find the best location in the timeline"""
        best_interval = max_group_info.interval

        # This repeats the calculation from find_max_group, pass this instead?
        # time_remaining = group_data.group.exec_time() - group_data.group.total_used()  # clock time
        # n_time_remaining = time2slots(self.time_slot_length, time_remaining)  # number of time slots
        # n_min, n_slots_remaining = self._min_slots_remaining(max_group_info.group_data.group)

        if max_group_info.n_slots_remaining < len(max_group_info.interval):
            # Determine position based on max integrated score
            # If we don't end up here, then the group will have to be split later
            best_interval = GreedyMaxOptimizer._integrate_score(night_idx, max_group_info)

        return best_interval

    def nir_slots(self, science_obs, n_slots_filled, len_interval) -> Tuple[int, int, ObservationID]:
        """
        Return the starting and ending timeline slots (indices) for the NIR science observations.
        """

        # science, split at atom
        slot_start_nir = None
        slot_end_nir = None
        slot_start = 0
        obs_id_nir = None
        for obs in science_obs:
            obs_id = obs.id
            cumul_seq = obs.cumulative_exec_times()
            atom_start = self._first_nonzero_time_idx(cumul_seq)
            atom_end = atom_start

            n_slots_acq = time2slots(self.time_slot_length, obs.acq_overhead)
            visit_length = n_slots_acq + time2slots(self.time_slot_length, cumul_seq[atom_end])

            # TODO: can this be done w/o a loop? convert cumm_seq to slots, and find the value that fits
            while n_slots_filled + visit_length <= len_interval and atom_end <= len(cumul_seq) - 2:
                atom_end += 1
                visit_length = n_slots_acq + time2slots(self.time_slot_length, cumul_seq[atom_end])

            slot_end = slot_start + visit_length - 1
            # NIR science time for to determine the number of tellurics
            if any(inst in obs.required_resources() for inst in ObservatoryProperties.nir_instruments()):
                if slot_start_nir is None:
                    slot_start_nir = slot_start + n_slots_acq  # start of the science sequence, after acq
                slot_end_nir = slot_end
                obs_id_nir = obs_id

            n_slots_filled += visit_length

            slot_start = slot_end + 1  # for the next iteration

        return slot_start_nir, slot_end_nir, obs_id_nir

    def mean_airmass(self, obs_id: ObservationID, interval: Interval, night_idx: NightIndex) -> npt.NDArray[float]:
        """
        Calculate the mean airmass of an observation over the given interval
        """
        programid = obs_id.program_id()
        #     print(obsid.id, programid.id)
        airmass = self.selection.program_info[programid].target_info[obs_id][night_idx].airmass[interval]

        return np.mean(airmass)

    def place_standards(self,
                        night_idx: NightIndex,
                        interval: Interval,
                        science_obs: List[Observation],
                        partner_obs: List[Observation],
                        n_std: int,
                        verbose: bool = False) -> Tuple[List[Observation], List[bool]]:
        """
        Pick the standards that best match the NIR science observations by airmass
        """

        standards = []
        placement = []

        if verbose:
            print('Running place_standards')

        xdiff_min = xdiff_before_min = xdiff_after_min = 99999.
        std_before = None
        std_after = None
        # If only one standard needed, try before or after, use best airmass match
        # TODO: Any preference to taking the standard before or after?
        # TODO: Check scores to confirm that the observations are schedulable (?)
        for partcal_obs in partner_obs:
            # Need the length of the calibration sequence only
            n_slots_cal = time2slots(self.time_slot_length, partcal_obs.exec_time())
            n_slots_acq = time2slots(self.time_slot_length, partcal_obs.acq_overhead)

            # Try std first
            # Mean std airmass
            slot_start = n_slots_acq
            slot_end = n_slots_cal - 1

            xmean_cal = self.mean_airmass(partcal_obs.id, interval[slot_start:slot_end + 1], night_idx=night_idx)

            if verbose:
                print(f'Standard {partcal_obs.id.id}')
                print(f'\t n_slots_acq = {n_slots_acq}, n_slots_cal = {n_slots_cal}')
                print(f'\t slot_start = {slot_start} slot_end = {slot_end}')
                print(f'\t interval[slot_start] = {interval[slot_start]} interval[slot_end] = {interval[slot_end]}')
                print('\t Try std before')
                print(f'\t length = {len(interval[slot_start:slot_end + 1])} xmean_cal_before = {xmean_cal}')

            if self.show_plots:
                self.plot_airmass(partcal_obs.id, interval=interval[slot_start:slot_end + 1], night_idx=night_idx)

            # Mean NIR science airmass
            idx_start_nir, idx_end_nir, obs_id_nir = self.nir_slots(science_obs, n_slots_cal, len(interval))
            slot_start_nir = slot_end + idx_start_nir
            slot_end_nir = slot_end + idx_end_nir

            xmean_nir = self.mean_airmass(obs_id_nir, interval[slot_start_nir:slot_end_nir + 1], night_idx=night_idx)
            xdiff_before = np.abs(xmean_nir - xmean_cal)

            if verbose:
                print(f'NIR science (after std) {obs_id_nir.id}')
                print(f'\t idx_start_nir = {idx_start_nir} idx_end_nir = {idx_end_nir}')
                print(f'\t slot_start_nir = {slot_start_nir} slot_end_nir = {slot_end_nir}')
                print(f'\t interval[slot_start_nir] = {interval[slot_start_nir]} interval[slot_end_nir] = {interval[slot_end_nir]}')
                print(f'\t first index interval: {interval[0]} last index interval: {interval[-1]}')
                print(f'\t {interval[slot_start_nir:slot_end_nir + 1]}')
                print(f'\t length = {len(interval[slot_start_nir:slot_end_nir + 1])} xmean_nir = {xmean_nir}')
                print(f'\t xdiff_before = {xdiff_before}')

            if self.show_plots:
                self.plot_airmass(obs_id_nir, interval=interval[slot_start_nir:slot_end_nir + 1], night_idx=night_idx)

            # Try std last
            # Mean std airmass
            len_int = len(interval)
            slot_start = len_int - 1 - n_slots_cal + n_slots_acq
            slot_end = slot_start + n_slots_cal - n_slots_acq - 1

            xmean_cal = self.mean_airmass(partcal_obs.id, interval[slot_start:slot_end + 1], night_idx=night_idx)

            if verbose:
                print('\n\t Try std after')
                print(f'\t slot_start = {slot_start} slot_end = {slot_end} len_int = {len_int}')
                print(f'\t {len(interval[slot_start:slot_end + 1])} xmean_cal_after = {xmean_cal}')

            if self.show_plots:
                self.plot_airmass(partcal_obs.id, interval=interval[slot_start:slot_end + 1], night_idx=night_idx)

            # Mean NIR science airmass
            slot_start_nir = idx_start_nir
            slot_end_nir = idx_end_nir

            xmean_nir = self.mean_airmass(obs_id_nir, interval[slot_start_nir:slot_end_nir + 1], night_idx=night_idx)
            xdiff_after = np.abs(xmean_nir - xmean_cal)

            if verbose:
                print(f'NIR science (before std) {obs_id_nir.id}')
                print(f'\t slot_start_nir = {slot_start_nir} slot_end_nir = {slot_end_nir}')
                print(f'\t {len(interval[slot_start_nir:slot_end_nir + 1])} xmean_nir = {xmean_nir}')
                print(f'\t xdiff_after = {xdiff_after}')

            if self.show_plots:
                self.plot_airmass(obs_id_nir, interval=interval[slot_start_nir:slot_end_nir + 1], night_idx=night_idx)

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

        if n_std > 1:
            placement = [True, False]
            standards = [std_before, std_after]

        return standards, placement
    
    @staticmethod
    def _charge_time(observation: Observation, atom_start: int = 0, atom_end: int = -1) -> None:
        """Pseudo (internal to GM) time accounting, or charging.
           GM must assume that each scheduled observation is executed and then adjust the completeness fraction
           and scoring accordingly. This does not update the database or Collector"""
        seq_length = len(observation.sequence)

        if atom_end < 0:
            atom_end += seq_length

        # Update observation status
        if atom_end == seq_length - 1:
            observation.status = ObservationStatus.OBSERVED
        else:
            observation.status = ObservationStatus.ONGOING

        for n_atom in range(atom_start, atom_end + 1):
            # "Charge" the expected program and partner times for the atoms:
            observation.sequence[n_atom].program_used = observation.sequence[n_atom].prog_time
            observation.sequence[n_atom].partner_used = observation.sequence[n_atom].part_time

            # Charge the acq to the first atom based on observation class
            if n_atom == atom_start:
                if observation.obs_class == ObservationClass.PARTNERCAL:
                    observation.sequence[n_atom].partner_used += observation.acq_overhead
                elif observation.obs_class == ObservationClass.SCIENCE or \
                        observation.obs_class == ObservationClass.PROGCAL:
                    observation.sequence[n_atom].program_used += observation.acq_overhead

            # For completeness
            observation.sequence[n_atom].observed = True
            observation.sequence[n_atom].qa_state = QAState.PASS

    def plot_airmass(self,
                     obs_id: ObservationID,
                     interval: Optional[Interval] = None,
                     night_idx: NightIndex = 0) -> None:
        """
        Plot airmass vs time slot.
        """
        programid = obs_id.program_id()

        airmass = self.selection.program_info[programid].target_info[obs_id][night_idx].airmass

        x = np.array([i for i in range(len(airmass))], dtype=int)
        p = plt.plot(airmass)
        colour = p[-1].get_color()
        if interval is not None:
            plt.plot(x[interval], airmass[interval], color=colour, linewidth=4)
        plt.ylim(2.5, 0.95)
        plt.title(obs_id.id)
        plt.xlabel('Time Slot')
        plt.ylabel('Airmass')
        plt.show()

    @staticmethod
    def _plot_interval(score: NightTimeslotScores,
                       interval: Interval,
                       best_interval: Interval,
                       label: str = "") -> None:
        """Plot score vs time_slot for the time interval under consideration"""

        # score = group_data.group_info.scores[night]
        x = np.array([i for i in range(len(score))], dtype=int)
        p = plt.plot(x, score)
        colour = p[-1].get_color()
        if best_interval is not None:
            plt.plot(x[best_interval], score[best_interval], color=colour, linewidth=4)
        # ylim = ax.get_ylim()
        # xlim = ax.get_xlim()
        plt.axvline(interval[0], ymax=1.0, color='black')
        plt.axvline(interval[-1], ymax=1.0, color='black')
        # plt.plot([iint[0], iint[0] + nttime], [0, 0], linewidth=6)
        # plt.plot([iint[0], iint[0] + nmin], [0, 0], linewidth=2)
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Time Slot', fontsize=12)
        if label != '':
            plt.title(label)
        plt.show()

    def plot_timelines(self, night_idx: NightIndex = 0, alt: bool = False) -> None:
        """Airmass and Score vs time/slot plot of the timelines for a night"""

        for site in self.sites:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
            # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
            obs_order = self.timelines[night_idx][site].get_observation_order()
            for idx, istart, iend in obs_order:
                if idx != -1:
                    unique_group_id = self.obs_group_ids[idx]
                    obs_id = ObservationID(unique_group_id.id)
                    program_id = obs_id.program_id()
                    scores = self.selection.program_info[program_id].group_data_map[unique_group_id]. \
                        group_info.scores[night_idx]
                    if alt:
                        y = self.selection.program_info[program_id].target_info[obs_id][night_idx].alt.degree
                    else:
                        y = self.selection.program_info[program_id].target_info[obs_id][night_idx].airmass
                    x = np.array([i for i in range(len(y))], dtype=int)
                    p = ax1.plot(x, y)
                    ax2.plot(x, np.log10(scores))

                    colour = p[-1].get_color()
                    ax1.plot(x[istart:iend + 1], y[istart:iend + 1], linewidth=4, color=colour,
                             label=obs_id.id)
                    ax2.plot(x[istart:iend + 1], np.log10(scores[istart:iend + 1]), linewidth=4, color=colour,
                             label=obs_id.id)

            # ax1.plot(self.timelines[night][site].time_slots)
            # ax1.axhline(0.0, xmax=1.0, color='black')
            if alt:
                ax1.axhline(30.0, xmax=1.0, color='black')
                ax1.set_ylim(15, 95)
                ax1.set_ylabel('Altitude [deg]')
            else:
                ax1.axhline(2.0, xmax=1.0, color='black')
                ax1.set_ylim(2.5, 0.95)
                ax1.set_ylabel('Airmass')
            ax1.set_xlabel('Time Slot')
            ax1.set_title(f"Night {night_idx + 1}: {site.name}")
            ax1.legend()

            ax2.set_xlabel('Time Slot')
            ax2.set_ylabel('log(Score)')
            ax2.set_title(f"Night {night_idx + 1}: {site.name}")
            # ax2.legend()

            plt.show()

    def _update_score(self, program: Program, night_idx: NightIndex) -> None:
        """Update the scores of the incomplete groups in the scheduled program"""

        if self.verbose:
            print(f"Starting _update_score for {program.id.id}")

        program_calculations = self.selection.score_program(program)

        if program_calculations is not None:
            # print("Re-score incomplete schedulable_groups")
            for unique_group_id in program_calculations.top_level_groups:
                if self.verbose:
                    print(f"\tRescoring {unique_group_id.id}")
                group_data = program_calculations.group_data_map[unique_group_id]
                group, group_info = group_data
                # Trap any key errors to prevent crashes, but this prevents the rescoring needed
                try:
                    schedulable_group = self.selection.schedulable_groups[unique_group_id]
                except KeyError:
                    logger.error(f'Schedulable_group key error for {unique_group_id.id}')
                    return None
                # print(f"{unique_group_id.id} {schedulable_group.group.exec_time()} {schedulable_group.group.total_used()}")
                if self.verbose:
                    print(f"\t\tNew max score: {np.max(group_info.scores[night_idx]):8.3f}")
                # update scores in schedulable_groups if the group is not completely observed
                if schedulable_group.group.exec_time() >= schedulable_group.group.total_used():
                    schedulable_group.group_info.scores = group_info.scores
                    # schedulable_group.group_info.scores[:] = group_info.scores[:]
                # print(f"\tUpdated max score: {np.max(schedulable_group.group_info.scores[night_idx]):7.2f}")


    def _update_group_list(self, max_group_info: MaxGroup, night_idx: NightIndex):
        """Remove groups as needed from group_data_list. Navigates the group tree to evaluate completion,
           sets the next timing window for cadences, and re-scores the program."""

        def sep(indent: int) -> str:
            return '-----' * indent

        def set_next_active(next_group: Group, timing_window) -> bool:
            """Set the next group to active, recursively follow down the group gree as needed,
               end set new the timing window for cadences (non-consecutive AND groups"""

            next_group.active = True
            # print(f"Setting {next_group.unique_id} to active with timing window")
            # print(f"{timing_window[0].iso} {timing_window[1].iso}")
            if next_group.group_option in [AndOption.CONSEC_ORDERED, AndOption.CONSEC_ANYORDER]:
                obs = next_group.observations()[0]
                site = obs.site
                # print(f"observation: {obs.id.id}")
                # Clock times for each time slot
                times = self.selection.night_events[site].times[night_idx]
                # print(f"{times[0]} {times[-1]}")
                # print(f"{tw_exclude_idx}")
                if next_group.unique_id in self.selection.schedulable_groups.keys():
                    target_info = p.target_info[obs.id]
                    schedulable_group = self.selection.schedulable_groups[next_group.unique_id]
                    vis_idx = target_info[night_idx].visibility_slot_idx
                    tw_exclude_idx = np.where(
                        np.logical_or(times[vis_idx] < timing_window[0],
                                      times[vis_idx] > timing_window[1])
                    )[0]
                    tw_include_idx = np.where(
                        np.logical_and(times[vis_idx] >= timing_window[0],
                                      times[vis_idx] <= timing_window[1])
                    )[0]
                    # print(f"Max score before excluding tw: {np.max(schedulable_group.group_info.scores[night_idx])}")
                    # Set scores to 0 outsize of the timing windows
                    schedulable_group.group_info.scores[night_idx][vis_idx[tw_exclude_idx]] = 0.0
                    # print(f"Max score after excluding tw: {np.max(schedulable_group.group_info.scores[night_idx])}")
                    # print(target_info[night_idx].visibility_slot_idx)
                    # print(tw_include_idx)
                    # Update target visibilities for future re-scores
                    target_info[night_idx].visibility_slot_idx = vis_idx[tw_include_idx]
                    # print(target_info[night_idx].visibility_slot_idx)
            set_next = False
            # If the next group is a custom group, set the first child active. if OR, set all children active.
            if next_group.group_option in [AndOption.CUSTOM, AndOption.NONE]:
                for child in next_group.children:
                    if child.previous_id == GROUP_NONE_ID:
                        set_next = set_next_active(child, timing_window)
                        if next_group.group_option == AndOption.CUSTOM:
                            break
            return set_next

        def trim_tree(group: Group, depth: int = 1) -> None:
            """Trim the unneeded branches from the tree"""

            # print(f"{sep(depth)} {group.id.id}")
            if group.unique_id in self.group_ids:
                # self.group_data_list.remove(group)
                # print(f"{sep(depth)} update_group_list: removing {group.id.id}")
                group_data = self.selection.schedulable_groups[group.unique_id]
                if group_data in self.group_data_list:
                    self.group_data_list.remove(group_data)
                    # print(f"{sep(depth)} update_group_list: removing {group.unique_id}")
            if not group.is_observation_group():
                for child in group.children:
                    trim_tree(child)

        def traverse_group_tree(prog_info: ProgramInfo, group: Group, end_time: datetime, depth: int = 1,
                                set_next: bool = True) -> None:
            """Traverse parent groups (up, then down each branch) starting at the given group"""

            # print(f"Group {group.id} with parent {group.parent_id}, AndOption={group.group_option}")
            if group.group_option in [AndOption.NONE, AndOption.ANYORDER, AndOption.CUSTOM]:
                group.number_observed += 1
            # print(f"{sep(depth)} group {group.unique_id}: to_observe {group.number_to_observe}, "
            #       f"observed {group.number_observed}")
            if group.number_observed == group.number_to_observe:
                # Traverse down the lower branches
                # print(f"{sep(depth)} group complete, trim the tree")
                trim_tree(group, depth=depth+1)

            # If not at the root group, get parent info and move up the tree
            if group.parent_id != GROUP_NONE_ID:
                # Parent group
                parent_unique_id = unique_group_id(prog_info.program.id, group.parent_id)
                parent = prog_info.program.get_group(parent_unique_id)

                if parent.group_option == AndOption.CUSTOM and group.next_id != GROUP_NONE_ID and set_next:
                    next_group = prog_info.program.get_group(unique_group_id(prog_info.program.id, group.next_id))
                    # print(f"{parent.id.id} {end_time.astimezone(tz=timezone.utc)} delay_min = {parent.delay_min}, "
                    #       f"delay_max={parent.delay_max}")
                    timing_window = [Time(end_time + parent.delay_min), Time(end_time + parent.delay_max)]
                    # print(f"Timing window: [{timing_window[0].iso}, {timing_window[1].iso}]")
                    set_next = set_next_active(next_group, timing_window)

                # subgroup = prog_info.group_data_map[parent]
                traverse_group_tree(prog_info, parent, end_time, set_next=set_next)

        # Get program information
        p = self.selection.program_info[max_group_info.group_data.group.program_id]
        # print(f"\tgroup_data_map: {p.group_data_map.keys()}")
        # p.program.show()

        # Update scores
        self._update_score(p.program, night_idx=night_idx)

        # Traverse and trim tree
        traverse_group_tree(p, max_group_info.group_data.group, max_group_info.end_time)



    def _run(self, plans: Plans) -> None:

        # Fill plans for all sites on one night
        while not self.timelines[plans.night_idx].all_done() and len(self.group_data_list) > 0:

            # print(f"\nNight {plans.night_idx + 1}")

            # Find the group with the max score in an open interval
            max_group_info = self._find_max_group(plans)

            # If something found, add it to the timeline and plan
            if max_group_info is not None:
                # max_score, max_group, max_interval = max_group_info
                added = self.add(plans.night_idx, max_group_info)
                if added:
                    if self.verbose:
                        print(f"Group {max_group_info.group_data.group.unique_id.id} with "
                              f"max score {max_group_info.max_score} added.")
                    # Clean up group_data_list and rescore
                    self._update_group_list(max_group_info, plans.night_idx)
            else:
                # Nothing remaining can be scheduled
                # for plan in plans:
                #     plan.is_full = True
                # TODO NOTE: Does this really mean the timeline is full?
                for timeline in self.timelines[plans.night_idx]:
                    logger.warning(f'Setting timelines corresponding to {plans.night_idx} to full (no max_group_info).')
                    timeline.is_full = True

        if self.show_plots:
            self.plot_timelines(plans.night_idx)

        # Write observations from the timelines to the output plan
        self.output_plans(plans)

    def _length_visit(self, t_acq, t_seq):
        """Calculate the number of time slots in a visit.
           Avoid rounding errors by summing times before determining the number of slots"""
        return time2slots(self.time_slot_length, (t_acq + t_seq))

    def _add_visit(self,
                   night_idx: NightIndex,
                   obs: Observation,
                   max_group_info: GroupData | MaxGroup,
                   best_interval: Interval,
                   n_slots_filled: int) -> int:
        """
        Add an observation to the timeline and do pseudo-time accounting.
        Returns the number of time slots filled.
        """

        verbose = False

        site = max_group_info.group_data.group.observations()[0].site
        timeline = self.timelines[night_idx][site]
        # program = self.selection.program_info[max_group_info.group_data.group.program_id].program

        # print(self.obs_group_ids)
        # print(obs.id, obs.to_unique_group_id)
        iobs = self.obs_group_ids.index(obs.to_unique_group_id)
        cumul_seq = obs.cumulative_exec_times()

        atom_start = self._first_nonzero_time_idx(cumul_seq)
        atom_end = atom_start

        # n_slots_acq = time2slots(self.time_slot_length, obs.acq_overhead)
        if verbose:
            print(f'_add_visit for {obs.unique_id.id}: {n_slots_filled} {atom_start} {atom_end}')
            for next_atom in range(len(cumul_seq)):
                print(next_atom, n_slots_filled + self._length_visit(obs.acq_overhead, cumul_seq[next_atom]))

        # type inspector cannot infer that cumul_seq[idx] is a timedelta.
        # noinspection PyTypeChecker
        visit_length = self._length_visit(obs.acq_overhead, cumul_seq[atom_end])
        next_atom = atom_end + 1
        # TODO: review the following logic with split sequences in GPP
        while (next_atom <= len(cumul_seq) - 1 and
               n_slots_filled + self._length_visit(obs.acq_overhead, cumul_seq[next_atom]) <= len(best_interval)):
            atom_end += 1
            # noinspection PyTypeChecker
            visit_length = self._length_visit(obs.acq_overhead, cumul_seq[atom_end])
            next_atom = atom_end + 1

        n_slots_filled += visit_length

        # add to timeline (time_slots)
        start_time_slot, start = timeline.add(iobs, visit_length, best_interval)

        # Get visit score and store information for the output plans
        end_time_slot = start_time_slot + visit_length - 1
        visit_score = sum(max_group_info.group_data.group_info.scores[night_idx][start_time_slot:end_time_slot + 1])
        peak_score = max(max_group_info.group_data.group_info.scores[night_idx][start_time_slot:end_time_slot + 1])

        self.obs_in_plan[site][start_time_slot] = ObsPlanData(
            obs=obs,
            obs_start=start,
            obs_len=visit_length,
            atom_start=atom_start,
            atom_end=atom_end,
            visit_score=visit_score,
            peak_score=peak_score,
        )

        # pseudo (internal) time charging
        # print(f"{program.id.id}: ")
        # print(f"before charge_time total_used: {program.total_used()} program_used: {program.program_used()}")
        self._charge_time(obs, atom_start=atom_start, atom_end=atom_end)
        # print(f"after  charge_time total_used: {program.total_used()} program_used: {program.program_used()}")

        return n_slots_filled, start

    def add(self, night_idx: NightIndex, max_group_info: GroupData | MaxGroup) -> bool:
        """
        Add a group to a Plan - find the best location within the interval (maximize the score) and select standards
        """

        # TODO: update base method?
        # Add method should handle those
        standards: List[Observation] = []

        # This is where we'll split groups/observations and integrate under the score
        # to place the group in the timeline

        site = max_group_info.group_data.group.observations()[0].site
        timeline = self.timelines[night_idx][site]
        result = False
        start_time = None # datetime
        total_slots_filled = 0
        n_std_placed = 0

        if self.verbose:
            print(f"Greedymax.add group {max_group_info.group_data.group.unique_id.id} "
                  f"with max score{max_group_info.max_score:8.4f}")
            print(f"\tTimeline slots remaining = {timeline.slots_unscheduled()}, "
                  f"n_slots_remaining = {max_group_info.n_slots_remaining}")
            print(f"\tNumber to observe={max_group_info.group_data.group.number_to_observe}, "
                  f"number observed = {max_group_info.group_data.group.number_observed}, "
                  f"n_std = {max_group_info.n_std}")
            print(f"\tInterval start end: {max_group_info.interval[0]} {max_group_info.interval[-1]}")

        if not timeline.is_full:
            # Find the best location in timeline for the group
            best_interval = self._find_group_position(night_idx, max_group_info)

            if self.show_plots:
                self._plot_interval(max_group_info.group_data.group_info.scores[night_idx], max_group_info.interval,
                                    best_interval,
                                    label=f'Night {night_idx + 1}: {max_group_info.group_data.group.unique_id.id}')

            # When/If we eventually support splitting NIR observations, then we need to calculate the
            # NIR science time in best_interval and the number of basecal (e.g. telluric standards) needed.
            # This may require the calibration service for selecting standards.
            # For now, we assume that NIR observations are not split, and we use the telluric standards provided.

            # Pick standard(s) if needed
            n_slots_cal = 0
            before_std = None
            after_std = None
            prog_obs = max_group_info.group_data.group.program_observations()
            part_obs = max_group_info.group_data.group.partner_observations()

            # print(f'Adding {max_group_info.group_data.group.unique_id.id} MaxScore:{max_group_info.max_score:7.2f} '
            #       f'n_std:{max_group_info.n_std} obs_mode:{max_group_info.group_data.group.obs_mode()} '
            #       f'nir_sci:{max_group_info.exec_sci_nir}')
            # print(f'{max_group_info.group_data.group.required_resources()}')
            # [print(wav) for wav in max_group_info.group_data.group.wavelengths()]
            # print('Program observations')
            # [print(f'\t {obs.unique_id.id}') for obs in prog_obs]
            # print('Partner observations')
            # [print(f'\t {obs.unique_id.id}') for obs in part_obs]
            if max_group_info.n_std > 0:
                if max_group_info.exec_sci_nir > ZeroTime:
                    standards, place_before = self.place_standards(night_idx, best_interval, prog_obs, part_obs,
                                                                   max_group_info.n_std, verbose=self.verbose)
                    for ii, std in enumerate(standards):
                        n_slots_cal += time2slots(self.time_slot_length, std.exec_time())
                        # print(f"{std.id.id} {place_before[ii]} {n_slots_cal}")
                        if place_before[ii]:
                            before_std = std
                        else:
                            after_std = std
                else:
                    prog_obs.extend(part_obs)

            n_slots_filled = 0
            if before_std is not None:
                obs = before_std
                # print(f"Adding before_std: {obs.to_unique_group_id} {obs.id.id}")
                n_slots_filled, start = self._add_visit(night_idx, obs, max_group_info, best_interval, n_slots_filled)
                max_group_info.group_data.group.number_observed += 1
                n_std_placed += 1
                start_time = start
                total_slots_filled += n_slots_filled


            # split at atoms
            for obs in prog_obs:
                # Reserve space for the cals, otherwise the science observes will fill the interval
                n_slots_filled = n_slots_cal
                # print(f"Adding science: {obs.to_unique_group_id} {obs.id.id}")
                n_slots_filled, start = self._add_visit(night_idx, obs, max_group_info, best_interval, n_slots_filled)
                start_time = start if start_time is None else start_time
                # ToDo: eventually check whether any are split, for now we consider it observed
                #  to avoid multiple visits on one night
                max_group_info.group_data.group.number_observed += 1
                if after_std is not None:
                    # "put back" time for the final standard
                    n_slots_filled -= time2slots(self.time_slot_length, standards[-1].exec_time())
                total_slots_filled += n_slots_filled

            if after_std is not None:
                obs = after_std
                # print(f"Adding after_std: {obs.to_unique_group_id} {obs.id.id}")
                n_slots_filled, start = self._add_visit(night_idx, obs, max_group_info, best_interval, n_slots_filled)
                max_group_info.group_data.group.number_observed += 1
                n_std_placed += 1
                total_slots_filled += n_slots_filled


            # If group is not split, inactivate any unused standards
            if max_group_info.n_slots_remaining == n_slots_filled:
                for obs in part_obs:
                    if obs not in standards:
                        obs.status = ObservationStatus.INACTIVE

            # Update the number to observe for the number of standards
            # print(f"greedymax.add: number_to_observe changed from {max_group_info.group_data.group.number_to_observe} "
            #       f"to {len(prog_obs) + max_group_info.n_std} for {max_group_info.group_data.group.id.id}")
            max_group_info.group_data.group.number_to_observe = len(prog_obs) + n_std_placed
            max_group_info.start_time = start_time
            max_group_info.end_time = start_time + total_slots_filled * self.time_slot_length

            # TODO: Shift to remove any gaps in the plan?

            if timeline.slots_unscheduled() <= 0:
                logger.warning(f'Timeline for {night_idx} is full: no slots remain unscheduled.')
                timeline.is_full = True

            result = True

        return result

    def output_plans(self, plans: Plans) -> None:
        """Write visit information from timelines to output plans, ensures chronological order"""

        # print(f'output_plans')

        for timeline in self.timelines[plans.night_idx]:
            obs_order = timeline.get_observation_order()
            for idx, start_time_slot, end_time_slot in obs_order:
                if idx > -1:
                    # obs_id = self.obs_group_ids[idx]
                    # TODO: HACK. I don't see an easy way around this since an obs_group_id has no idea it is an
                    # TODO: UniqueGroupID of an observation group. Maybe we can make obs_in_plan a map from
                    # TODO: UniqueGroupID to... something, instead of an ObservationID to an ObsPlanData.
                    # print('output_plans')
                    # print(f'idx = {idx} len(obs_group_ids) = {len(self.obs_group_ids)}')
                    # print(self.obs_group_ids)

                    # Add visit to final plan
                    obs_in_plan = self.obs_in_plan[timeline.site][start_time_slot]
                    # print(f'{obs_in_plan.obs.id.id:20} {start_time_slot:4} {end_time_slot:4} {end_time_slot - start_time_slot + 1:4} '
                    #       f'{obs_in_plan.obs_len:4}')
                    plans[timeline.site].add(obs_in_plan.obs,
                                             obs_in_plan.obs_start,
                                             obs_in_plan.atom_start,
                                             obs_in_plan.atom_end,
                                             start_time_slot,
                                             obs_in_plan.obs_len,
                                             obs_in_plan.visit_score,
                                             obs_in_plan.peak_score)
                    plans[timeline.site].update_time_slots(timeline.slots_unscheduled())
            # print('')
