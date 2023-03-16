# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from datetime import datetime, timedelta
# from typing import Mapping

# from lucupy.minimodel.program import ProgramID

from scheduler.core.calculations.selection import Selection
from scheduler.core.calculations import GroupData, ProgramInfo
from scheduler.core.plans import Plan, Plans
from scheduler.core.components.optimizer.timeline import Timelines
from .base import BaseOptimizer

import numpy as np
# import astropy.units as u


class GreedyMaxOptimizer(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits for the rest of the night in a greedy fashion.
    """

    def __init__(self):
        self.groups = []
        self.group_ids = []
        # self.obs_groups = []     # remove if not used
        self.obs_group_ids = []
        self.timelines = []
        self.sites = []
        self.min_visit_len = timedelta(minutes=30)

    def setup(self, selection: Selection) -> GreedyMaxOptimizer:
        """
        Preparation for the optimizer e.g. create chromosomes, etc.
        """
        self.groups = list(selection.schedulable_groups.values())
        self.group_ids = list(selection.schedulable_groups)
        for gid, group_data in selection.schedulable_groups.items():
            if group_data.group.is_observation_group():
                self.obs_group_ids.append(group_data.group.unique_id())
                # self.obs_groups.append(group_data.group)
            elif group_data.group.is_scheduling_group():
                for subgroup in group_data.group.children:
                    if subgroup.is_observation_group():
                        self.obs_group_ids.append(subgroup.unique_id())
                        # self.obs_groups.append(subgroup)

        period = len(list(selection.night_events.values())[0].time_grid)
        print('Period: ', period)
        self.timelines = [Timelines(selection.night_events, night) for night in range(period)]
        self.sites = list(selection.night_events.keys())

        return self

    @staticmethod
    def _allocate_time(plan: Plan, obs_time: timedelta) -> datetime:
        """
        Allocate time for an observation inside a Plan
        This should be handled by the optimizer as can vary from algorithm to algorithm
        """
        # Get first available slot
        start = plan.start
        if len(plan.visits) > 0:
            start = plan.visits[-1].start_time + plan.visits[-1].time_slots * plan.time_slot_length

        return start

    def _find_max_group(self, plans: Plans):
        """Find the group with the max score in an open interval"""

        # the number of time slots in the minimum visit length
        time_slot_length = plans.plans[self.sites[0]].time_slot_length
        n_min_visit = int(np.ceil(self.min_visit_len / time_slot_length))

        # If true just analyze the only first open interval, like original GM, eventually make a parameter or setting
        only_first_interval = False

        # Get the unscheduled, available intervals (time slots)
        open_intervals = {}
        for site in self.sites:
            open_intervals[site] = self.timelines[plans.night][site].get_available_intervals(only_first_interval)
            # print(site, open_intervals[site])

        maxscores = []
        groups = []
        intervals = []  # interval indices
        n_times_remaining = []
        # ids = []  # group index for the scores
        ii = 0    # groups index counter
        # Make a list of scores in the remaining groups
        for group in self.groups:
            site = group.group.observations()[0].site
            if not plans[site].is_full:
                for iint, interval in enumerate(open_intervals[site]):
                    # print(f'Interval: {iint}')
                    smax = np.max(group.group_info.scores[plans.night][interval])
                    if smax > 0.0:
                        # Check if the interval is long enough to be useful (longer than min visit length)
                        # Remaining time for the group
                        # also should see if it can be split, for now we assume that all can be
                        time_remaining = group.group.exec_time() - group.group.total_used()  # clock time
                        n_time_remaining = int(np.ceil((time_remaining / time_slot_length))) # number of time slots

                        # Short groups should be done entirely, update the min useful time
                        # is the extra variable needed, or just modify n_min_visit?
                        # n_min = n_min_visit
                        # if n_time_remaining - n_min <= n_min:
                        #     n_min = n_time_remaining
                        # Until we support splitting, just use the remaining time
                        n_min = n_time_remaining

                        # Need to evaluate sub-intervals (e.g. timing windows)

                        # Compare interval length with remaining group length
                        if n_min <= len(interval):
                            maxscores.append(smax)
                            # ids.append(ii)         # needed?
                            groups.append(group)
                            intervals.append(iint)
                            n_times_remaining.append(n_time_remaining)
                            ii += 1

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
            site = groups[iscore_sort[ii]].group.observations()[0].site
            max_interval = open_intervals[site][intervals[iscore_sort[ii]]]

            # Prefer a group in the allowed score range if it does not require splitting,
            # otherwise take the top scorer
            selected = False
            while not selected and ii < len(iscore_sort):
                site = groups[iscore_sort[ii]].group.observations()[0].site
                if maxscores[iscore_sort[ii]] >= score_limit and \
                        n_times_remaining[iscore_sort[ii]] <= len(open_intervals[site][intervals[iscore_sort[ii]]]):
                    max_score = maxscores[iscore_sort[ii]]
                    max_group = groups[iscore_sort[ii]]
                    max_interval = open_intervals[site][intervals[iscore_sort[ii]]]
                    selected = True
                ii += 1

        return max_score, max_group, max_interval

    def _run(self, plans: Plans):

        # Fill plans for all sites on one night
        while not plans.all_done() and len(self.groups) > 0:

            print(f"\nNight {plans.night + 1}")

            # Find the group with the max score in an open interval
            max_score, max_group, max_interval = self._find_max_group(plans)

            # If something found, add it to the timeline and plan
            if max_interval is not None:
                added = self.add(max_group, plans, max_interval)
                if added:
                    print(f'{max_group.group.unique_id()} with max score {max_score} added.')
                    self.groups.remove(max_group)  # should really only do this if all time used (not split)
            else:
                # Nothing remaining can be scheduled
                for plan in plans:
                    plan.is_full = True
                for timeline in self.timelines[plans.night]:
                    timeline.is_full = True

    def add(self, group: GroupData, plans: Plans, interval) -> bool:
        """
        Add a group to a Plan
        """
        # TODO: Missing different logic for different AND/OR GROUPS
        # Add method should handle those

        # This is where we'll split groups/observations and integrate under the score
        # to place the group in the timeline

        site = group.group.observations()[0].site
        plan = plans[site]
        if not plan.is_full:
            for observation in group.group.observations():
                if observation not in plan:
                    # add to plan
                    obs_len = plan.time2slots(observation.exec_time())
                    # add to timeline (time_slots)
                    iobs = self.obs_group_ids.index(observation.id)  # index in observation list
                    start = self.timelines[plans.night][site].add(iobs, obs_len, interval)
                    # Put the timelines call in _allocate_time, or use that for time accounting updates?
                    # start = self._allocate_time(plan, observation.exec_time())

                    # Add visit to final plan - in general won't be in chronological order
                    # Maybe add this as a final step once GM is finished?
                    plan.add(observation, start, obs_len)
                    # Where to do time accounting? Here, _allocate_time or in plan/timelines.add?

            if plan.time_left() <= 0:
                plan.is_full = True

            return True
        else:
            return False
