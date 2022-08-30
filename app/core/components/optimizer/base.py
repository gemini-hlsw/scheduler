from abc import ABC, abstractmethod
from ...calculations.groupinfo import GroupData
from ...plans import Plans
from lucupy.minimodel.program import ProgramID
from ...calculations.programinfo import ProgramInfo
from typing import Mapping, List


class BaseOptimizer(ABC):
    """
    Base class for all Optimizer components.
    Each optimizing algorithm needs to implement the following methods:

    schedule: method that triggers the formation of the plan
    setup: method that prepares the algorithm to be used for scheduling
    add: method that adds a group to a plan
    _run: main driver for the algorithm

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
    def add(self, group: GroupData, Plans: Plans):
        ...
