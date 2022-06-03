from abc import ABC, abstractmethod
from common.minimodel.group import Group
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
    def schedule(self, nights: List[Plans]):
        for plans in nights:
            self._run(plans)
    
    @abstractmethod
    def _run(self, plans: Plans):
        ...

    @abstractmethod
    def setup(self, program_info: Mapping[ProgramID, ProgramInfo]):
        ...

    @abstractmethod
    def add(self, group: Group):
        ...
