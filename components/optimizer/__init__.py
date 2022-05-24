from .greedymax import GreedyMax
from common.minimodel import Plans


class Optimizer:
    """
    Entrypoint to interact with an BaseOptimizer object.
    All algorithms need to follow the same structure to create a Plan
    """

    def __init__(self, selection, algorithm=GreedyMax(some_parameter=1)):
        self.algorithm = algorithm.add(selection)
    
    def schedule(self, night_events) -> Plans:
        # TODO: This forces to make Plans for each site SEPARATELY.
        # This would create and issue with OR groups as observations can be schedule 
        # on different sites but not twice. Old GM had some check-ins to handle this 
        # cases but right now we don't have OR groups implemented.
        plans = Plans(night_events)
        for plan in plans:
            self.algorithm.schedule(plan)
        return plans
