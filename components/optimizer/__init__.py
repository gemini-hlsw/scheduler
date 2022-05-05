from .greedymax import GreedyMax


class Optimizer:
    """
    Factory class for creating optimizers.
    """

    def __init__(self, selection, algorithm=GreedyMax(some_parameter=1)):
        self.algorithm = algorithm.add(selection)
    
    def schedule(self):
        self.algorithm.schedule()
