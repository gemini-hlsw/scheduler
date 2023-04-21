# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, FrozenSet, List, Optional, Tuple

from scheduler.core.calculations.selection import Selection
from scheduler.core.calculations import GroupData
from scheduler.core.plans import Plan, Plans
from scheduler.core.components.optimizer.timeline import Timelines
from .base import BaseOptimizer
from . import Interval

from lucupy.minimodel import Group, Observation, ObservationID, Site, UniqueGroupID
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ObsPlanData:
    """
    Storage to conglomerate observation information for the plan together.
    """
    obs: Observation
    obs_start: datetime
    obs_len: int
    visit_score: float


class GreedyMaxOptimizer(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits for the rest of the night in a greedy fashion.
    """

    def __init__(self, min_visit_len: timedelta = timedelta(minutes=30), show_plots: bool = False):
        self.group_data_list: List[GroupData] = []
        self.group_ids: List[UniqueGroupID] = []
        self.obs_group_ids: List[UniqueGroupID] = []
        self.timelines: List[Timelines] = []
        self.sites: FrozenSet[Site] = frozenset()
        self.obs_in_plan: Dict[ObservationID, ObsPlanData] = {}
        self.min_visit_len = min_visit_len
        self.show_plots = show_plots
        self.time_slot_length: Optional[timedelta] = None

    def setup(self, selection: Selection) -> GreedyMaxOptimizer:
        """
        Preparation for the optimizer.
        """
        self.group_ids = list(selection.schedulable_groups)
        self.group_data_list = list(selection.schedulable_groups.values())
        # self._process_group_data(self.group_data_list)
        self.obs_group_ids = list(selection.obs_group_ids) # noqa
        num_nights = selection.num_nights
        # print('Number of nights: ', num_nights)
        self.timelines = [Timelines(selection.night_events, night) for night in range(num_nights)]
        self.sites = selection.sites
        self.time_slot_length = selection.time_slot_length
        return self

    @staticmethod
    def non_zero_intervals(scores: npt.NDArray[float]) -> npt.NDArray[int]:
        """
        Calculate the non-zero intervals in the data.
        This consists of an array with entries of the form [a, b] where
        the non-zero interval runs from a (inclusive) to b (exclusive).
        See test_greedymax.py for an example.
        """
        # Create an array that is 1 where the score is greater than 0, and pad each end with an extra 0.
        not_zero = np.concatenate(([0], np.greater(scores, 0), [0]))
        abs_diff = np.abs(np.diff(not_zero))

        # Return the ranges for each nonzero interval.
        return np.where(abs_diff == 1)[0].reshape(-1, 2)

    def _min_slots_remaining(self, group: Group) -> Tuple[int, int]:
        """
        Returns the minimum number of time slots for the remaining time.
        Right now, this is the same value twice.
        TODO: When group splitting is supported, it will be minimum number and remaining number of time slots.
        """

        # the number of time slots in the minimum visit length
        # min_visit_timeslots = int(np.ceil(self.min_visit_len / self.time_slot_length))
        # print(f"n_min_visit: {n_min_visit}")

        # Calculate the remaining clock time necessary for the group to be complete.
        time_remaining = group.exec_time() - group.total_used()

        # Calculate the number of time slots needed to complete the group.
        # n_slots_remaining = int(np.ceil((time_remaining / self.time_slot_length)))  # number of time slots
        # This use of time2slots works but is probably not kosher, need to make this more general
        n_slots_remaining = Plan.time2slots(self.time_slot_length, time_remaining)

        # Short groups should be done entirely, update the min useful time
        # is the extra variable needed, or just modify n_min_visit?
        # n_min = n_min_visit
        # if n_time_remaining - n_min <= n_min:
        #     n_min = n_time_remaining

        # Until we support splitting, just use the remaining time.
        n_min = n_slots_remaining

        return n_min, n_slots_remaining

    def _find_max_group(self, plans: Plans) -> Optional[Tuple[float, GroupData, Interval]]:
        """
        Find the group with the max score in an open interval
        Returns None if there is no such group.
        Otherwise, returns the score, group_data, and interval.
        """

        # If true just analyze the only first open interval, like original GM, eventually make a parameter or setting
        only_first_interval = False

        # Get the unscheduled, available intervals (time slots)
        open_intervals = {site: self.timelines[plans.night][site].get_available_intervals(only_first_interval)
                          for site in self.sites}

        max_scores = []
        groups = []
        intervals = []  # interval indices
        n_times_remaining = []
        # ids = []  # group index for the scores
        # ii = 0    # groups index counter

        # Make a list of scores in the remaining groups
        for group_data in self.group_data_list:
            site = group_data.group.observations()[0].site
            if not self.timelines[plans.night][site].is_full:
                for interval_idx, interval in enumerate(open_intervals[site]):
                    # print(f'Interval: {iint}')
                    # scores = group_data.group_info.scores[plans.night]

                    # Get the maximum score over the interval.
                    smax = np.max(group_data.group_info.scores[plans.night][interval])
                    if smax > 0.0:
                        # Check if the interval is long enough to be useful (longer than min visit length).
                        # Remaining time for the group.
                        # Also, should see if it can be split.
                        n_min, num_time_slots_remaining = self._min_slots_remaining(group_data.group)

                        # Evaluate sub-intervals (e.g. timing windows, gaps in the score).
                        # Find time slot locations where the score > 0.
                        # interval is a numpy array that indexes into the scores for the night to return a sub-array.
                        check_interval = group_data.group_info.scores[plans.night][interval]
                        group_intervals = GreedyMaxOptimizer.non_zero_intervals(check_interval)
                        max_score_on_interval = 0.0
                        max_interval = None
                        for group_interval in group_intervals:
                            grp_interval_length = group_interval[1] - group_interval[0]

                            max_score = np.max(group_data.group_info.scores[plans.night]
                                               [interval[group_interval[0]:group_interval[1]]])

                            # Find the max_score in the group intervals with non-zero scores
                            # The length of the non-zero interval must be at least as large as
                            # the minimum length
                            if max_score > max_score_on_interval and grp_interval_length >= n_min:
                                max_score_on_interval = max_score
                                max_interval = group_interval

                        if max_interval is not None:
                            max_scores.append(max_score_on_interval)
                            # ids.append(ii)         # needed?
                            groups.append(group_data)
                            # intervals.append(interval_idx)
                            # print(max_interval)
                            # print(interval[max_interval[0]:max_interval[1]])
                            intervals.append(interval[max_interval[0]:max_interval[1]])
                            n_times_remaining.append(num_time_slots_remaining)
                        # ii += 1

        max_score: Optional[float] = None
        max_group: Optional[GroupData] = None
        max_interval: Optional[Interval] = None
        if len(max_scores) > 0:
            # sort scores from high to low
            iscore_sort = np.flip(np.argsort(max_scores))
            ii = 0

            max_score = max_scores[iscore_sort[ii]]  # maximum score in time interval
            # consider groups with max scores within frac_score_limit of max_score
            # Only highest score: frac_score_limit = 0.0
            # Top 10%: frac_score_limit = 0.1
            # Consider everything: frac_score_limit = 1.0
            frac_score_limit = 1.0
            score_limit = max_score * (1.0 - frac_score_limit)
            max_group = groups[iscore_sort[ii]]
            max_interval = intervals[iscore_sort[ii]]

            # Prefer a group in the allowed score range if it does not require splitting,
            # otherwise take the top scorer
            selected = False
            while not selected and ii < len(iscore_sort):
                if (max_scores[iscore_sort[ii]] >= score_limit and
                        n_times_remaining[iscore_sort[ii]] <= len(intervals[iscore_sort[ii]])):
                    max_score = max_scores[iscore_sort[ii]]
                    max_group = groups[iscore_sort[ii]]
                    max_interval = intervals[iscore_sort[ii]]
                    selected = True
                ii += 1

        if max_score is None or max_group is None or max_interval is None:
            return None
        return max_score, max_group, max_interval

    def _integrate_score(self,
                         group_data: GroupData,
                         interval: Interval,
                         group_time_slots: int,
                         night_idx: int) -> Interval:
        """Use the score array to find the best location in the timeline

            group_data: Group data of group with maximum score
            interval: the timeline interval, where to place group_data
            n_time: length of the group in time steps
            night: night counter
        """
        start = interval[0]
        end = interval[-1]
        scores = group_data.group_info.scores[night_idx]
        max_integral_score = scores[0]

        if len(interval) > 1:
            # Slide across the interval, integrating the score over the group length
            for idx in range(interval[0], interval[-1] - group_time_slots + 2):

                integral_score = sum(scores[idx:idx + group_time_slots])

                if integral_score > max_integral_score:
                    max_integral_score = integral_score
                    start = idx
                    end = start + group_time_slots - 1

        # print(f"Initial start end: {start} {end} {n_time} {end - start + 1}")

        # Shift to window boundary if within minimum block time of edge.
        # If near both boundaries, choose boundary with higher score.
        score_start = scores[start]  # score at start
        score_end = scores[end-1]  # score at end
        delta_start = start - interval[0]  # difference between start of window and block
        delta_end = interval[-1] - end  # difference between end of window and block
        n_min, n_time_remaining = self._min_slots_remaining(group_data.group)
        # print(f"delta_start: {delta_start}, delta_end: {delta_end}")
        # print(f"score_start: {score_start}, score_end: {score_end}")
        if delta_start < n_min and delta_end < n_min:
            if score_start > score_end and score_start > 0.0:
                # print('a')
                start = interval[0]
                end = start + group_time_slots - 1
            elif score_end > 0.0:
                # print('b')
                start = interval[-1] - group_time_slots + 1
                end = interval[-1]
        elif delta_start < n_min and score_start > 0.0:
            # print('c')
            start = interval[0]
            end = start + group_time_slots - 1
        elif delta_end < n_min and score_end > 0:
            # print('d')
            start = interval[-1] - group_time_slots + 1
            end = interval[-1]

        # print(f"Shifted start end: {start} {end} {end - start + 1}")

        # Make final list of indices for the highest scoring shifted sub-interval
        best_interval = np.arange(start=start, stop=end+1)
        # print(f"len(best_interval): {len(best_interval)}")

        return best_interval

    def _find_group_position(self,
                             group_data: GroupData,
                             interval: Interval,
                             night_idx: int) -> Interval:
        """Find the best location in the timeline"""
        best_interval = interval

        # # This repeats the calculation from find_max_group, pass this?
        # time_remaining = group_data.group.exec_time() - group_data.group.total_used()  # clock time
        # # This is the same as time2slots, need to make that more generally available
        # n_time_remaining = int(np.ceil((time_remaining / self.time_slot_length)))  # number of time slots
        n_min, n_time_remaining = self._min_slots_remaining(group_data.group)

        if n_time_remaining < len(interval):
            # Determine position based on max integrated score
            # If we don't end up here, then the group will have to be split later
            best_interval = self._integrate_score(group_data, interval, n_time_remaining, night_idx)

        return best_interval

    @staticmethod
    def _plot_interval(score, interval, best_interval, label: str = "") -> None:
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

    def plot_timelines(self, night) -> None:
        """Score vs time/slot plot of the timelines for a night"""

        # This may need to be moved out of here to access scores and airmasses

        # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        for site in self.sites:
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
            # obs_order = self.timelines[night][site].get_observation_order()
            # for idx, istart, iend in obs_order:
            #     obsid = 'Unscheduled'
            #     if idx != -1:
            #         obsid = self.obs_group_ids[idx]
            #         p = ax1.plot(self.obs_groups[idx].group_info.scores[night])
            #         colour = p[-1].get_color()
            #         ax1.plot(self.obs_groups[idx].scores[night][istart:iend + 1], linewidth=4,
            #                  label=obsid)
            #     print(idx, obsid, istart, iend)

            ax1.plot(self.timelines[night][site].time_slots)
            ax1.axhline(0.0, xmax=1.0, color='black')
            ax1.set_title(f"Night {night + 1}: {site.name}")
            # ax1.legend()
            plt.show()

    def _run(self, plans: Plans) -> None:

        # Fill plans for all sites on one night
        while not self.timelines[plans.night].all_done() and len(self.group_data_list) > 0:

            print(f"\nNight {plans.night + 1}")

            # Find the group with the max score in an open interval
            max_data = self._find_max_group(plans)

            # If something found, add it to the timeline and plan
            if max_data is not None:
                max_score, max_group, max_interval = max_data
                added = self.add(max_group, plans.night, max_interval)
                if added:
                    print(f'{max_group.group.unique_id} with max score {max_score} added.')
                    self.group_data_list.remove(max_group)  # should really only do this if all time used (not split)
            else:
                # Nothing remaining can be scheduled
                # for plan in plans:
                #     plan.is_full = True
                for timeline in self.timelines[plans.night]:
                    timeline.is_full = True

        if self.show_plots:
            self.plot_timelines(plans.night)

        # TODO: Write observations from the timelines to the output plan
        self.output_plans(plans)

    def add(self, group_data: GroupData, night: int, interval: Optional[Interval] = None) -> bool:
        """
        Add a group to a Plan
        """
        # TODO: Missing different logic for different AND/OR GROUPS
        # Add method should handle those

        # This is where we'll split groups/observations and integrate under the score
        # to place the group in the timeline

        site = group_data.group.observations()[0].site
        timeline = self.timelines[night][site]
        result = False
        if not timeline.is_full:
            # Find the best location in timeline for the group
            best_interval = self._find_group_position(group_data, interval, night)
            # print(f"Interval start end: {interval[0]} {interval[-1]}")
            # print(f"Best interval start end: {best_interval[0]} {best_interval[-1]}")

            if self.show_plots:
                GreedyMaxOptimizer._plot_interval(group_data.group_info.scores[night], interval, best_interval,
                                                  label=f'Night {night + 1}: {group_data.group.unique_id}')

            for observation in group_data.group.observations():
                print(f"**** {self.obs_group_ids}, {observation.id}")
                # iobs = self.obs_group_ids.index(observation.id)  # index in observation list
                # TODO: HACK
                unique_group_id = UniqueGroupID(observation.id.id)
                iobs = self.obs_group_ids.index(unique_group_id)

                # if iobs not in timeline.time_slots:  # when splitting it could appear multiple times
                # Calculate the length of the observation (visit)
                time_remaining = observation.exec_time() - observation.total_used()
                # This use of time2slots works but is probably not kosher, need to make this more general
                obs_len = Plan.time2slots(self.time_slot_length, time_remaining)

                # add to timeline (time_slots)
                start_time_slot, start = timeline.add(iobs, obs_len, best_interval)
                # Put the timelines call in _allocate_time, or use that for time accounting updates?
                # start = self._allocate_time(plan, observation.exec_time())

                # Sergio's Note:
                # Both of these lines are added to calculate NightStats. This could be modified,
                # as in calculated somewhere else or in a different way, but are needed when plan.add is called.
                # In the future we could merge this with timeline but the design on that is TBD.
                # TODO: Partner calibrations should not contribute to this
                visit_score = sum(group_data.group_info.scores[night][start_time_slot:start_time_slot + obs_len])

                self.obs_in_plan[observation.id] = ObsPlanData(
                    obs=observation,
                    obs_start=start,
                    obs_len=obs_len,
                    visit_score=visit_score
                )

            if timeline.slots_unscheduled() <= 0:
                timeline.is_full = True

            result = True

        return result

    def output_plans(self, plans: Plans) -> None:
        """Write visit information from timelines to output plans, ensures chronological order"""

        for timeline in self.timelines[plans.night]:
            obs_order = timeline.get_observation_order()
            for idx, start_time_slot, end_time_slot in obs_order:
                if idx > -1:
                    # obs_id = self.obs_group_ids[idx]
                    # TODO: HACK
                    unique_group_id = self.obs_group_ids[idx]
                    obs_id = ObservationID(unique_group_id.id)

                    # Add visit to final plan
                    obs_in_plan = self.obs_in_plan[obs_id]
                    plans[timeline.site].add(obs_in_plan.obs,
                                             obs_in_plan.obs_start,
                                             start_time_slot,
                                             obs_in_plan.obs_len,
                                             obs_in_plan.visit_score)
