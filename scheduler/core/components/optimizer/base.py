# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from typing import List, Mapping, Optional

from lucupy.minimodel.program import ProgramID

from scheduler.core.calculations.groupinfo import GroupData
from scheduler.core.calculations.programinfo import ProgramInfo
from scheduler.core.plans import Plans
from . import Interval


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
    # def setup(self, selection: Selection):
    def setup(self, program_info: Mapping[ProgramID, ProgramInfo]):
        ...

    @abstractmethod
    def add(self, group: GroupData, plans: Plans, interval: Optional[Interval]):
        ...
