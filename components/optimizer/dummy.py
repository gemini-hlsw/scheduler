from components.optimizer.base import BaseOptimizer
from common.minimodel.plan import Plans, Plan
from common.minimodel.program import ProgramID
from common.minimodel.group import Group
from common.calculations.programinfo import ProgramInfo
from datetime import datetime
from typing import Mapping
import random


class DummyOptimizer(BaseOptimizer):

    def __init__(self, seed=42):
        # Set seed for replication
        random.seed(seed)
        self.programs = []

    def _allocate_time(self, plan: Plan) -> datetime:
        """
        Allocate time for an observation inside a Plan
        This should be handle by the optimizer as can vary from algorithm to algorithm
        """
        # Get first available slot
        start = plan.start
        for v in plan._visits:
            start += v.start_time
        return start

    def _run(self, plans: Plans):
        """
        Gives a random group/observation to add to plan
        """

        while plans.all_done():
            groups = []
            for program in self.programs:
                groups.extend(program.group_data.values())
            ran_group = random.choice(groups)
            if self.add(ran_group):
                print('group added')
            else:
                print('group not added')
        
    def setup(self, programInfo: Mapping[ProgramID, ProgramInfo]):
        """
        Preparation for the optimizer i.e create chromosomes, etc.
        """
        self.programs = [p for p in programInfo.values()]
        return self
    
    def add(self, group: Group, plans: Plans) -> bool:
        """
        Add a group to a Plan
        This is called when a new group is added to the program
        """                        
        # TODO: Missing different logic for different AND/OR GROUPS
        # Add method should handle those
        for observation in group.group.observations():
            plan = plans[observation.site]
            if not plan.is_full and plan.site == observation.site:
                obs_len = plan.time2slots(observation.total_used())
                if (plan.time_left() >= obs_len):
                    start = self._allocate_time(plan, observation.total_used())
                    plan.add(observation, start, obs_len)
                    return True
                else:
                    # TODO: DO a partial insert
                    # Splitting groups is not yet implemented
                    # Right now we are just going to finish the plan
                    plan.is_full = True
                    return False
