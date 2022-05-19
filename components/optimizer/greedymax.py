from common.minimodel import Group
from .base import BaseOptimizer
from typing import Dict
from common.minimodel import Group, Plan
from components.selector import GroupInfo

Selection = Dict[Group, GroupInfo]


class GreedyMax(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits in a greedy fashion.
    """

    def __init__(self, some_parameter=1):
        # Parameters specifically for the GreedyMax optimizer should be go here
        self.some_parameter = some_parameter
        self._visits = []
        self.selection = None
    
    def _find_max_group(self):
        """
        Find the group with the highest score.
        """
        # TODO: This makes a case for adding Info to the group
        info = [g[1] for g in self.selection.values()]
        return max(i.scores[0][0] for i in info)  # This just plain wrong and for demo purposes
 
    def _insert(self, max_group: Group, plan: Plan):
        """
        Insert the group into the plan.
        This might be general for all optimizers but for GreedyMax we need to
        move things around as we insert (also splitting). It could be force to implement by
        the abstract class
        """
        plan.add_group(max_group)
        return True

    def _run(self, plan: Plan):
        """
        GreedyMax logic goes here.
        """
        max_group = None  # OR Visit
        iter = 0
        scheduled = False

        while not scheduled:
            iter += 1

            max_group = self._find_max_group()
            if self._insert(max_group, plan):
                scheduled = True
            else:
                pass
                # TODO: this would span a infinite loop if the group is not inserted
                # and find_max_group keeps returning the same group. If I recall correctly
                # scores are set to 0.0 and the group is not inserted.
        
    def add(self, selection: Selection):
        # Preparation for the optimizer i.e create chromosomes, etc.
        self.selection = selection
        return self

    def schedule(self):
        """
        Entry point for any optimizer. This method is given by the base class.
        """
        return super().schedule()

    def get_visits(self):
        return self._visits
