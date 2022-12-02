# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

from lucupy.minimodel.program import ProgramID

from app.core.calculations import GroupData, ProgramInfo
from app.core.plans import Plan, Plans
from .base import BaseOptimizer

import numpy as np
import astropy.units as u


class GreedyMaxOptimizer(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits in a greedy fashion.
    """

    def __init__(self, seed=42):
        self.groups = []

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
        # for v in plan.visits:
        #     # delta = v.start_time - start + obs_time
        #     start = v.start_time + v.time_slots * plan.time_slot_length
        return start


    def setup(self, program_info: Mapping[ProgramID, ProgramInfo]) -> GreedyMaxOptimizer:
        """
        Preparation for the optimizer e.g. create chromosomes, etc.
        """
        self.groups = []
        # self.scores = []
        for p in program_info.values():
            self.groups.extend([g for g in p.group_data.values() if g.group.is_observation_group()])
            # Suggestion from Seb
            # self.groups.extend([g for g in p.group_data.values() if all(x.is_observation_group() for x in g.group.children)])
            # self.scores.extend([s for s in p.group_data])
        return self

    def _run(self, plans: Plans):

        print()
        while not plans.all_done() and len(self.groups) > 0:
            # for plan in plans:
            #     print(plan.site, plan.is_full)
        # if len(self.groups) > 0:

            print('Night ', plans.night + 1)
            # just take the first group available
            # group = self.groups[0]
            # if self.add(group, plans):
            #     # TODO: All observations in the group are being inserted so the whole group
            #     # can be removed
            #     self.groups.remove(group)
            # else:
            #     print('group not added')

            max_score = 0.0         # maximum score in time interval
            max_group = {}
            maxscores = []
            groups = []
            ids = [] # group index for the scores
            ii = 0
            # Collect scores
            for group in self.groups:
                site = group.group.observations()[0].site
                if not plans[site].is_full:
                    smax = np.max(group.group_info.scores[plans.night])
                    maxscores.append(smax)
                    ids.append(ii)
                    groups.append(group)
                    ii += 1
                    # print(group.group.id, group.group.exec_time(), smax)

            # sort
            jj = np.flip(np.argsort(maxscores))
            for ii in range(len(jj)):
                max_score = maxscores[jj[ii]]
                max_group = groups[jj[ii]]
                # for obs in group.group.observations():
                #     print(f'\t {obs.id} {obs.exec_time()}')
                #     for atom in obs.sequence:
                #         print(f'\t\t {atom.id} {atom.exec_time}')
                if max_score > 0.0:
                    if self.add(max_group, plans):
                        # TODO: All observations in the group are being inserted so the whole group
                        # can be removed
                        print(f'{max_group.group.id} with max score {max_score} added.')
                        self.groups.remove(max_group)
                        break
                    else:
                        # print('group not added')
                        print(f'{max_group.group.id} with max score {max_score} not added.')
                else:
                    # # Finished, nothing more to schedule
                    # for plan in plans:
                    #     plan.is_full = True
                    break
            print('')

    def add(self, group: GroupData, plans: Plans) -> bool:
        """
        Add a group to a Plan
        This is called when a new group is added to the program
        """
        # TODO: Missing different logic for different AND/OR GROUPS
        # Add method should handle those
        for observation in group.group.observations():
            plan = plans[observation.site]
            if not plan.is_full:
                obs_len = plan.time2slots(observation.exec_time())
                print(plan.time_left(), obs_len, observation.exec_time())
                if (plan.time_left() >= obs_len) and not plan.has(observation):
                    start = self._allocate_time(plan, observation.exec_time())
                    plan.add(observation, start, obs_len)
                    return True
                else:
                    # TODO: DO a partial insert
                    # Splitting groups is not yet implemented
                    # Right now we are just going to finish the plan
                    plan.is_full = True
                    print(plan.site, plan.is_full)
                    return False
