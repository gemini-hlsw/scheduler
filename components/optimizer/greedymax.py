from common.minimodel import Group
from .base import BaseOptimizer
from typing import Dict
from common.minimodel import Group
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
        return max(self.selection, key=lambda g: g.score)
    
    def _insert(self, max_group: Group):
        """
        Insert the group into the plan.
        This might be general for all optimizers but for GreedyMax we need to
        move things around as we insert (also splitting). It could be force to implement by
        the abstract class
        """
        return True

    def _run(self):
        """
        GreedyMax logic goes here.
        """
        max_group = None  # OR Visit
        iter = 0
        scheduled = False

        while not scheduled:
            iter += 1

            max_group = self._find_max_group()
            if self._insert(max_group):
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
