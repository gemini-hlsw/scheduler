from app.common.calculations.selection import Selection
from .greedymax import GreedyMax
from app.common.plans import Plans
from typing import List


class Optimizer:
    """
    Entrypoint to interact with an BaseOptimizer object.
    All algorithms need to follow the same structure to create a Plan
    """

    def __init__(self, selection: Selection, algorithm=None):
        self.algorithm = algorithm.setup(selection.program_info)
        self.night_events = selection.night_events
        # TODO: Assumes that all sites schedule the same amount of nights
        self.period = len(list(self.night_events.values())[0].time_grid)
    
    def schedule(self) -> List[Plans]:
        # Create set of plans for the amount of nights
        nights = [Plans(self.night_events, night) for night in range(self.period)]
        self.algorithm.schedule(nights)
        return nights
