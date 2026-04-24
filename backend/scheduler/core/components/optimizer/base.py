# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import final, List

from lucupy.types import Interval

from scheduler.core.calculations.groupinfo import GroupData
from scheduler.core.calculations.selection import Selection
from scheduler.core.plans import Plans


__all__ = [
    'BaseOptimizer',
    'MaxGroup',
]


@final
@dataclass(frozen=False)
class MaxGroup:
    """
    Store information about the selected group (max score)
    """
    group_data: GroupData
    max_score: float
    interval: Interval
    n_min: int
    n_slots_remaining: int
    n_std: int
    exec_sci_nir: timedelta
    start_time: datetime
    end_time: datetime


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
    def setup(self, selection: Selection):
        ...

    @abstractmethod
    def add(self, night: int, max_group_info: GroupData | MaxGroup):
        ...
