from abc import ABC, abstractmethod
from common.minimodel.plan import Plans, Plan
from common.minimodel.program import ProgramID
from common.calculations.programinfo import ProgramInfo
from typing import Mapping, Tuple, List


class BaseOptimizer(ABC):
    """
    Base class for all Optimizer components.
    Each optimizing algorithm needs to implement the following methods:

    schedule: methods that triggers the formation of the plan
    get_visits: methods that updates the plan

    """
    def schedule(self, plans: Plans):
        for night in range(len(plans.nights)):
            self._run(plans.nights[night])
    
    @abstractmethod
    # TODO: Replace to NightIndex
    def _run(self, plans: List[Plan]):
        ...

    @abstractmethod
    def add(self, program_info: Mapping[ProgramID, ProgramInfo]):
        ...
