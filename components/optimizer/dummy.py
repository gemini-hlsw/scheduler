from components.optimizer.base import BaseOptimizer, Selection
from common.minimodel import Plan
import random


class DummyOptimizer(BaseOptimizer):

    def __init__(self, seed=42):
        # Set seed for replication
        random.seed(seed)

    def _run(self, plan: Plan):
        """
        Gives a random group/observation to add to plan
        """
        ran_group = random.choice(list(self.selection.values()))
        # TODO: A useful thing might be transforming the total times to time slots
        # to reduce runtime calculations.
        group_length = plan._time2slots(ran_group[0].total_used())
        if (plan.time_slots_left() >= group_length):
            plan.add_group(ran_group)
        else:
            # TODO: DO a partial insert
            # Spliting groups is not yet implemented
            plan.add_group(ran_group)
        print(f'slots left in the plan: {plan.time_slots_left()}')
        
    def add(self, selection: Selection):
        # Preparation for the optimizer i.e create chromosomes, etc.
        self.selection = selection
        return self
