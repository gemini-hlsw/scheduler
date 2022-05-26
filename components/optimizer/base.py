from abc import ABC, abstractmethod
from common.minimodel import Group, Plan
from components.selector import GroupInfo, GroupID
from typing import Mapping, Tuple

Selection = Mapping[GroupID, Tuple[Group, GroupInfo]]


class BaseOptimizer(ABC):
    """
    Base class for all Optimizer components.
    Each optimizing algorithm needs to implement the following methods:

    schedule: methods that triggers the formation of the plan
    get_visits: methods that updates the plan

    """
    def schedule(self, plan: Plan):
        while not plan.is_full():
            self._run(plan)
    
    @abstractmethod
    def _run(self, plan: Plan):
        ...
