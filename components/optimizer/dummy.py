from components.optimizer.base import BaseOptimizer
from common.minimodel.plan import Plan
from common.minimodel.program import ProgramID
from common.calculations.programinfo import ProgramInfo
from datetime import datetime
from typing import List, Mapping
import random


class DummyOptimizer(BaseOptimizer):

    def __init__(self, seed=42):
        # Set seed for replication
        random.seed(seed)

    def allocate_time(self, plan: Plan) -> datetime:
        """
        Allocate time for an observation inside a Plan
        This should be handle by the optimizer as can vary from algorithm to algorithm
        """
        # Get first available slot
        start = plan.start
        for v in plan._visits:
            start += v.start_time
        return start

    def _run(self, plans: List[Plan]):
        """
        Gives a random group/observation to add to plan
        """
        while all(plan.is_full for plan in plans):
            for program in self.programs:
                ran_group = random.choice(program.group_data.values())
                for observation in ran_group.group.observations():
                    # TODO: This should be constant not lineal time
                    for plan in plans:
                        if not plan.is_full and plan.site == observation.site:
                            obs_len = plan.time2slots(observation.total_used())
                            if (plan.time_left() >= obs_len):
                                start = self.allocate_time(plan, observation.total_used())
                                plan.add(observation, start, obs_len)
                                break
                            else:
                                # TODO: DO a partial insert
                                # Splitting groups is not yet implemented
                                # Right now we are just going to finish the plan
                                plan.is_full = True
        
    def add(self, programInfo: Mapping[ProgramID, ProgramInfo]):
        # Preparation for the optimizer i.e create chromosomes, etc.
        self.programs = [p for p in programInfo.values()]
        return self
