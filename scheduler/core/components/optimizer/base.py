# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Mapping, List, Optional, Tuple

from lucupy.minimodel.program import ProgramID

from scheduler.core.calculations.groupinfo import GroupData
from scheduler.core.calculations.programinfo import ProgramInfo
from scheduler.core.plans import Plan, Plans

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
    def add(self, group: GroupData, plans: Plans, interval: Optional[Interval] = None):
        ...

    @staticmethod
    def _first_free_time(plan: Plan) -> Tuple[datetime, int]:
        """
        Get the first available start time and time slot in a Plan.
        """
        # Get first available slot
        if len(plan.visits) == 0:
            start = plan.start
            start_time_slot = 0
        else:
            start = plan.visits[-1].start_time + plan.visits[-1].time_slots * plan.time_slot_length
            start_time_slot = plan.visits[-1].start_time_slot + plan.visits[-1].time_slots + 1

        return start, start_time_slot
