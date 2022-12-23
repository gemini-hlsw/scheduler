# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from typing import FrozenSet, Mapping

from lucupy.minimodel import GroupID, Program, Observation, ObservationID

from .groupinfo import GroupDataMap
from .targetinfo import TargetInfoNightIndexMap


# Leave this as non-immutable, as program, group_data, and observations all might change.
@dataclass(frozen=True)
class ProgramInfo:
    """
    This represents the information for a program that contains schedulable components during the time frame
    under consideration, along with those schedulable components.
    """
    # Deep-copied.
    program: Program

    # Schedulable groups by ID and their information.
    group_data: GroupDataMap

    # Deep-coped.
    # Schedulable observations by their ID. This is duplicated in the group information above
    # but provided for convenience.
    observations: Mapping[ObservationID, Observation]

    # Immutable.
    # Target information relevant to this program.
    target_info: Mapping[ObservationID, TargetInfoNightIndexMap]

    # post-init members.
    observation_ids: FrozenSet[ObservationID] = field(init=False)
    group_ids: FrozenSet[GroupID] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'observation_ids', frozenset(self.observations.keys()))
        object.__setattr__(self, 'group_ids', frozenset(self.group_data.keys()))
