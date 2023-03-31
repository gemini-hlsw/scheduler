# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, NoReturn
# from typing import Mapping

# from lucupy.minimodel.program import ProgramID

from scheduler.core.calculations.selection import Selection
from scheduler.core.calculations import GroupData, ProgramInfo
from scheduler.core.plans import Plan, Plans
from scheduler.core.components.optimizer.timeline import Timelines
from .base import BaseOptimizer

import numpy as np
import matplotlib.pyplot as plt
# import astropy.units as u


class GreedyMaxOptimizer(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits for the rest of the night in a greedy fashion.
    """

    def __init__(self, min_visit_len: timedelta = timedelta(minutes=30), show_plots: bool = False):
        self.group_data_list = []
        self.group_ids = []
        # self.obs_groups = []     # remove if not used
        self.obs_group_ids = []
        self.timelines = []
        self.sites = []
        self.time_slot_length = timedelta
        self.min_visit_len = min_visit_len
        self.show_plots = show_plots

    def setup(self, selection: Selection) -> GreedyMaxOptimizer:
        """
        Preparation for the optimizer e.g. create chromosomes, etc.
        """
        self.group_ids = list(selection.schedulable_groups)
        self.group_data_list = list(selection.schedulable_groups.values())
        # self._process_group_data(self.group_data_list)
        self.obs_group_ids = list(selection.obs_group_ids)

        # As per my comment below: if you need period, it should be a member of Selection.
        # period = len(list(selection.night_events.values())[0].time_grid)
        num_nights = selection.plan_num_nights
        # print('Number of nights: ', num_nights)
        self.timelines = [Timelines(selection.night_events, night) for night in range(num_nights)]

        # As per my comment below.
        # self.sites = list(selection.night_events.keys())
        self.sites = selection.sites

        self.time_slot_length = selection.time_slot_length

        return self

    @staticmethod
    def _allocate_time(plan: Plan, obs_len: int) -> Tuple[datetime, int]:
        """
        Allocate time for an observation inside a Plan
        This should be handled by the optimizer as can vary from algorithm to algorithm
        """
        # Get first available slot
        start = plan.start
        start_time_slot = 0
        if len(plan.visits) > 0:
            start = plan.visits[-1].start_time + plan.visits[-1].time_slots * plan.time_slot_length
            start_time_slot = plan.visits[-1].start_time_slot + obs_len + 1

        return start, start_time_slot

    def non_zero_intervals(self, scores: np.ndarray) -> np.ndarray:

        # Create an array that is 1 where the score is greater than 0, and pad each end with an extra 0.
        isntzero = np.concatenate(([0], np.greater(scores, 0), [0]))
        absdiff = np.abs(np.diff(isntzero))
        # Get the ranges for each non zero interval
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

        return ranges

    def _min_slots_remaining(self, group_data) -> int:
        """Return the number of time slots for the remaining time"""

        # the number of time slots in the minimum visit length
        # time_slot_length = plans.plans[self.sites[0]].time_slot_length
        n_min_visit = int(np.ceil(self.min_visit_len / self.time_slot_length))
        # print(f"n_min_visit: {n_min_visit}")

        time_remaining = group_data.group.exec_time() - group_data.group.total_used()  # clock time
        # This is the same as time2slots
        n_slots_remaining = int(np.ceil((time_remaining / self.time_slot_length)))  # number of time slots

        # Short groups should be done entirely, update the min useful time
        # is the extra variable needed, or just modify n_min_visit?
        # n_min = n_min_visit
        # if n_time_remaining - n_min <= n_min:
        #     n_min = n_time_remaining
        # Until we support splitting, just use the remaining time
        n_min = n_slots_remaining

        return n_min, n_slots_remaining

    def _find_max_group(self, plans: Plans):
        """Find the group with the max score in an open interval"""

        # If true just analyze the only first open interval, like original GM, eventually make a parameter or setting
        only_first_interval = False

        # Get the unscheduled, available intervals (time slots)
        open_intervals = {site: self.timelines[plans.night][site].get_available_intervals(only_first_interval)
                          for site in self.sites}

        maxscores = []
        groups = []
        intervals = []  # interval indices
        n_times_remaining = []
        # ids = []  # group index for the scores
        # ii = 0    # groups index counter
        # Make a list of scores in the remaining groups
        for group_data in self.group_data_list:
            site = group_data.group.observations()[0].site
            if not plans[site].is_full:
                for interval_idx, interval in enumerate(open_intervals[site]):
                    # print(f'Interval: {iint}')
                    # scores = group_data.group_info.scores[plans.night]
                    smax = np.max(group_data.group_info.scores[plans.night][interval])
                    if smax > 0.0:
                        # Check if the interval is long enough to be useful (longer than min visit length)
                        # Remaining time for the group
                        # also should see if it can be split
                        n_min, n_time_remaining = self._min_slots_remaining(group_data)

                        # #valuate sub-intervals (e.g. timing windows, gaps in the score)
                        # Find ime slot locations where the score > 0
                        group_intervals = self.non_zero_intervals(group_data.group_info.scores[plans.night][interval])
                        max_score_on_interval = 0.0
                        max_interval = None
                        for group_interval in group_intervals:
                            grp_interval_length = group_interval[1] - group_interval[0]

                            max_score = np.max(group_data.group_info.scores[plans.night]
                                               [interval[group_interval[0]:group_interval[1]]])

                            # Find the max_sore in the group intervals with non-zero scores
                            # The length of the non-zero interval must be at least as large as
                            # the minimum length
                            if max_score > max_score_on_interval \
                                    and grp_interval_length >= n_min:
                                max_score_on_interval = max_score
                                max_interval = group_interval

                        if max_interval is not None:
                            maxscores.append(max_score_on_interval)
                            # ids.append(ii)         # needed?
                            groups.append(group_data)
                            # intervals.append(interval_idx)
                            # print(max_interval)
                            # print(interval[max_interval[0]:max_interval[1]])
                            intervals.append(interval[max_interval[0]:max_interval[1]])
                            n_times_remaining.append(n_time_remaining)
                        # ii += 1

        max_score = None
        max_group = None
        max_interval = None
        if len(maxscores) > 0:
            # sort scores from high to low
            iscore_sort = np.flip(np.argsort(maxscores))
            ii = 0

            max_score = maxscores[iscore_sort[ii]]  # maximum score in time interval
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
                if maxscores[iscore_sort[ii]] >= score_limit and \
                        n_times_remaining[iscore_sort[ii]] <= len(intervals[iscore_sort[ii]]):
                    max_score = maxscores[iscore_sort[ii]]
                    max_group = groups[iscore_sort[ii]]
                    max_interval = intervals[iscore_sort[ii]]
                    selected = True
                ii += 1

        return max_score, max_group, max_interval

    def _integrate_score(self, group_data, interval, n_time, night):
        """Use the score array to find the best location in the timeline

            group_data: Group data of group with maximum score
            interval: the timeline interval, where to place group_data
            n_time: length of the group in time steps
            night: night counter
        """
        best_interval = interval
        start = interval[0]
        end = interval[-1]
        scores = group_data.group_info.scores[night]
        max_integral_score = scores[0]

        if len(interval) > 1:
            # Slide across the interval, integrating the score over the group length
            for idx in range(interval[0], interval[-1] - n_time + 2):

                integral_score = sum(scores[idx:idx + n_time])

                if integral_score > max_integral_score:
                    max_integral_score = integral_score
                    start = idx
                    end = start + n_time - 1

        # print(f"Initial start end: {start} {end} {n_time} {end - start + 1}")

        # Shift to window boundary if within minimum block time of edge.
        # If near both boundaries, choose boundary with higher score.
        score_start = scores[start]  # score at start
        score_end = scores[end-1]  # score at end
        delta_start = start - interval[0]  # difference between start of window and block
        delta_end = interval[-1] - end # difference between end of window and block
        n_min, n_time_remaining = self._min_slots_remaining(group_data)
        # print(f"delta_start: {delta_start}, delta_end: {delta_end}")
        # print(f"score_start: {score_start}, score_end: {score_end}")
        if delta_start < n_min and delta_end < n_min:
            if score_start > score_end and score_start > 0.0:
                # print('a')
                start = interval[0]
                end = start + n_time - 1
            elif score_end > 0.0:
                # print('b')
                start = interval[-1] - n_time + 1
                end = interval[-1]
        elif delta_start < n_min and score_start > 0.0:
            # print('c')
            start = interval[0]
            end = start + n_time - 1
        elif delta_end < n_min and score_end > 0:
            # print('d')
            start = interval[-1] - n_time + 1
            end = interval[-1]

        # print(f"Shifted start end: {start} {end} {end - start + 1}")

        # Make final list of indices for the highest scoring shifted sub-interval
        best_interval = [ii for ii in range(start, end + 1)]
        # print(f"len(best_interval): {len(best_interval)}")

        return best_interval

    def _find_group_position(self, plan: Plan, group_data, interval, night):
        """Find the best location in the timeline"""
        best_interval = interval

        # # This repeats the calculation from find_max_group, pass this?
        # time_remaining = group_data.group.exec_time() - group_data.group.total_used()  # clock time
        # # This is the same as time2slots, need to make that more generally available
        # n_time_remaining = int(np.ceil((time_remaining / self.time_slot_length)))  # number of time slots
        n_min, n_time_remaining = self._min_slots_remaining(group_data)

        if n_time_remaining < len(interval):
            # Determine position based on max integrated score
            # If we don't end up here, then the group will have to be split later
            best_interval = self._integrate_score(group_data, interval, n_time_remaining, night)

        return best_interval

    def _plot_interval(self, score, interval, best_interval, label="") -> NoReturn:
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

    def plot_timelines(self, night) -> NoReturn:
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

    def _run(self, plans: Plans):

        # Fill plans for all sites on one night
        while not plans.all_done() and len(self.group_data_list) > 0:

            print(f"\nNight {plans.night + 1}")

            # Find the group with the max score in an open interval
            max_score, max_group, max_interval = self._find_max_group(plans)

            # If something found, add it to the timeline and plan
            if max_interval is not None:
                added = self.add(max_group, plans, max_interval)
                if added:
                    print(f'{max_group.group.unique_id()} with max score {max_score} added.')
                    self.group_data_list.remove(max_group)  # should really only do this if all time used (not split)
            else:
                # Nothing remaining can be scheduled
                for plan in plans:
                    plan.is_full = True
                for timeline in self.timelines[plans.night]:
                    timeline.is_full = True

        if self.show_plots:
            self.plot_timelines(plans.night)

        # TODO: Write observations from the timelines to the output plan
        # for timeline in self.timelines[plans.nights]:
        #     timeline.output_plan()

    def add(self, group_data: GroupData, plans: Plans, interval) -> bool:
        """
        Add a group to a Plan
        """
        # TODO: Missing different logic for different AND/OR GROUPS
        # Add method should handle those

        # This is where we'll split groups/observations and integrate under the score
        # to place the group in the timeline

        # TODO: switch to checking whether the timeline rather than the plan is full,
        # don't add observations to plan now since they will be out of chronological order
        site = group_data.group.observations()[0].site
        plan = plans[site]
        result = False
        if not plan.is_full:
            # Find the best location in timeline for the group
            best_interval = self._find_group_position(plan, group_data, interval, plans.night)
            # print(f"Interval start end: {interval[0]} {interval[-1]}")
            # print(f"Best interval start end: {best_interval[0]} {best_interval[-1]}")

            if self.show_plots:
                self._plot_interval(group_data.group_info.scores[plans.night], interval, best_interval,
                                    label=f'Night {plans.night + 1}: {group_data.group.unique_id()}')

            for observation in group_data.group.observations():
                if observation not in plan:
                    # add to plan
                    obs_len = plan.time2slots(observation.exec_time())

                    # Find the best location in the interval based on the score
                    # obs_len = len(best_interval)
                    # print(f"Inverval lengths: {len(interval)} {obs_len}")

                    # Allocate_time is only used for NightStats. See note
                    _, start_time_slot = GreedyMaxOptimizer._allocate_time(plan, obs_len)

                    # add to timeline (time_slots)
                    iobs = self.obs_group_ids.index(observation.id)  # index in observation list
                    start = self.timelines[plans.night][site].add(iobs, obs_len, best_interval)
                    # Put the timelines call in _allocate_time, or use that for time accounting updates?
                    # start = self._allocate_time(plan, observation.exec_time())

                    # Sergio's Note:
                    # Both of this lines are added to calculate NightStats, this could be modified, as in calculated somewhere
                    # else or in a different way. But are needed when plan.add is called.
                    # In the future we could merge this with timeline but the design on that is TBD.
                    visit_score = np.sum(group_data.group_info.scores[plans.night][start_time_slot:start_time_slot+obs_len])

                    # Add visit to final plan - in general won't be in chronological order
                    # Maybe add all observations as a final step once GM is finished?
                    plan.add(observation, start, start_time_slot, obs_len, visit_score)
                    # Where to do time accounting? Here, _allocate_time or in plan/timelines.add?

            if plan.time_left() <= 0:
                plan.is_full = True

            result = True

        return result
