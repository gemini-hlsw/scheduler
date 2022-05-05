from .base import BaseOptimizer


class GreedyMax(BaseOptimizer):
    """
    GreedyMax is an optimizer that schedules the visits in a greedy fashion.
    """

    def __init__(self, some_parameter=1):
        # Parameters specifically for the GreedyMax optimizer should be go here
        self.some_parameter = some_parameter
        self._visits = []  # TODO: This should go in base
        self.selection = None
        
    def add(self, selection):
        # Preparation for the optimizer i.e create chromosomes, etc.
        self.selection = selection
        return self

    def schedule(self):
        ...

    def get_visits(self):
        return self._visits
