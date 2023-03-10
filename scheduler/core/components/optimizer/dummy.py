# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import random
from datetime import datetime

from scheduler.core.calculations.selection import Selection
from scheduler.core.calculations import GroupData
from scheduler.core.plans import Plan, Plans
from .base import BaseOptimizer


class DummyOptimizer(BaseOptimizer):

    def __init__(self, seed=42):
        # Set seed for replication
        random.seed(seed)
        self.groups = []

    @staticmethod
    def _allocate_time(plan: Plan) -> datetime:
        """
        Allocate time for an observation inside a Plan
        This should be handled by the optimizer as can vary from algorithm to algorithm
        """
        # Get first available slot
        start = plan.start
        if len(plan.visits) > 0:
            start = plan.visits[-1].start_time + plan.visits[-1].time_slots * plan.time_slot_length
        # for v in plan.visits:
        #     delta = v.start_time - start + obs_time
        #     start += delta
        return start

    def _run(self, plans: Plans):
        """
        Gives a random group/observation to add to plan
        """

        while not plans.all_done() and len(self.groups) > 0:

            ran_group = random.choice(self.groups)
            if self.add(ran_group, plans):
                # TODO: All observations in the group are being inserted so the whole group
                # can be removed
                self.groups.remove(ran_group)
            else:
                print('group not added')

    def setup(self, selection: Selection) -> DummyOptimizer:
        """
        Preparation for the optimizer e.g. create chromosomes, etc.
        """
        self.groups = []
        for p in selection.program_info.values():
            self.groups.extend([g for g in p.group_data.values() if g.group.is_observation_group()])
        return self

    def add(self, group: GroupData, plans: Plans) -> bool:
        """
        Add a group to a Plan
        This is called when a new group is added to the program
        """
        # TODO: Missing different logic for different AND/OR GROUPS
        # Add method should handle those
        for observation in group.group.observations():
            plan = plans[observation.site]
            if not plan.is_full and plan.site == observation.site:
                obs_len = plan.time2slots(observation.exec_time())
                if plan.time_left() >= obs_len and observation not in plan:
                    start = DummyOptimizer._allocate_time(plan)
                    plan.add(observation, start, obs_len)
                    return True
                else:
                    # TODO: DO a partial insert
                    # Splitting groups is not yet implemented
                    # Right now we are just going to finish the plan
                    plan.is_full = True
                    return False
