# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import timedelta
from typing import FrozenSet, Mapping

from lucupy.minimodel import ProgramID, Site, UniqueGroupID

from .groupinfo import GroupData
from .nightevents import NightEvents
from .programinfo import ProgramInfo


@dataclass(frozen=True)
class Selection:
    """
    The selection of information passed by the Selector to the Optimizer.
    This includes the list of programs that are schedulable and the night event for the nights under consideration.
    """
    program_info: Mapping[ProgramID, ProgramInfo]
    night_events: Mapping[Site, NightEvents]
    schedulable_groups: Mapping[UniqueGroupID, GroupData]
    obs_group_ids: FrozenSet[UniqueGroupID]
    num_nights_plan: int
    time_slot_length: timedelta

    @property
    def sites(self) -> FrozenSet[Site]:
        return frozenset(self.night_events.keys())

    def __post_init__(self):
        object.__setattr__(self, 'program_ids', frozenset(self.program_info.keys()))
