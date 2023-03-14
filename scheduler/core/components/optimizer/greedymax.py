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
import astropy.units as u


class GreedyMaxOptimizer(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits for the rest of the night in a greedy fashion.
    """

    def __init__(self, seed=42):
        self.groups = []
        self.group_ids = []
        self.obs_groups = []
        self.obs_group_ids = []
        self.timelines = []

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

    def setup(self, selection: Selection) -> GreedyMaxOptimizer:
        """
        Preparation for the optimizer e.g. create chromosomes, etc.
        """
        self.groups = list(selection.schedulable_groups.values())
        self.group_ids = list(selection.schedulable_groups)
        for gid, group_data in selection.schedulable_groups.items():
            if group_data.group.is_observation_group():
                self.obs_group_ids.append(group_data.group.unique_id())
                self.obs_groups.append(group_data.group)
            elif group_data.group.is_scheduling_group():
                for subgroup in group_data.group.children:
                    if subgroup.is_observation_group():
                        self.obs_group_ids.append(subgroup.unique_id())
                        self.obs_groups.append(subgroup)
        # self.scores = []
        # for gid, group_data in selection.schedulable_groups.items():
        #     self.groups.append(group_data)
        period = len(list(selection.night_events.values())[0].time_grid)
        print('Period: ', period)
        self.timelines = [Timelines(selection.night_events, night) for night in range(period)]

        return self

    def _run(self, plans: Plans):

        print()
        # Fill all plans
        while not plans.all_done() and len(self.groups) > 0:

            print('Night ', plans.night + 1)
            maxscores = []
            groups = []
            ids = [] # group index for the scores
            ii = 0
            # Make a list of scores in the remaining groups
            for group in self.groups:
                site = group.group.observations()[0].site
                if not plans[site].is_full:
                    smax = np.max(group.group_info.scores[plans.night])
                    maxscores.append(smax)
                    ids.append(ii)
                    groups.append(group)
                    ii += 1
                    # print(group.group.id, group.group.exec_time(), smax)

            # sort scores from high to low
            max_score = 0.0  # maximum score in time interval
            max_group = {}
            jj = np.flip(np.argsort(maxscores))
            ii = 0
            added = False
            # Find the group with the maximum score that will fit in the remaining time
            # This will tend to avoid observation splitting, may want to limit to the band of the top-ranking group
            while not added and ii < len(jj):
                max_score = maxscores[jj[ii]]
                max_group = groups[jj[ii]]
                # for obs in group.group.observations():
                #     print(f'\t {obs.id} {obs.exec_time()}')
                #     for atom in obs.sequence:
                #         print(f'\t\t {atom.id} {atom.exec_time}')
                if max_score > 0.0:
                    # Try to add group
                    added = self.add(max_group, plans)
                    if added:
                        # TODO: All observations in the group are being inserted so the whole group
                        # can be removed from the active group list
                        print(f'{max_group.group.unique_id()} with max score {max_score} added.')
                        self.groups.remove(max_group)
                    else:
                        print(f'{max_group.group.unique_id()} with max score {max_score} not added.')
                else:
                    # Finished, nothing more to schedule
                    for plan in plans:
                        plan.is_full = True
                    for timeline in self.timelines[plans.night]:
                        timeline.is_full = True
                    break
                ii += 1
            if not added and ii == len(jj):
                # Nothing remaining can be scheduled
                for plan in plans:
                    plan.is_full = True
                for timeline in self.timelines[plans.night]:
                    timeline.is_full = True
            print('')

    def add(self, group: GroupData, plans: Plans) -> bool:
        """
        Add a group to a Plan
        """
        # TODO: Missing different logic for different AND/OR GROUPS
        # Add method should handle those
        site = group.group.observations()[0].site
        plan = plans[site]
        if not plan.is_full:
            grp_len = plan.time2slots(group.group.exec_time())
            print(plan.time_left(), grp_len, group.group.exec_time())
            if plan.time_left() >= grp_len:
                for observation in group.group.observations():
                    if observation not in plan:
                        # add to plan
                        obs_len = plan.time2slots(observation.exec_time())
                        start = self._allocate_time(plan, observation.exec_time())
                        plan.add(observation, start, obs_len)
                        # add to timeline (time slots)
                        iobs = self.obs_group_ids.index(observation.id)
                        self.timelines[plans.night][site].add(iobs, obs_len)
                return True
            else:
                return False
