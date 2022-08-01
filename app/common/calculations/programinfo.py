from dataclasses import dataclass
from typing import Mapping

from app.common.minimodel import Program, Observation, ObservationID
from .groupinfo import GroupDataMap
from .targetinfo import TargetInfoNightIndexMap


@dataclass(frozen=True)
class ProgramInfo:
    """
    This represents the information for a program that contains schedulable components during the time frame
    under consideration, along with those schedulable components.
    """
    program: Program

    # Schedulable groups by ID and their information.
    group_data: GroupDataMap

    # Schedulable observations by their ID. This is duplicated in the group information above
    # but provided for convenience.
    observations: Mapping[ObservationID, Observation]

    # Target information relevant to this program.
    target_info: Mapping[ObservationID, TargetInfoNightIndexMap]

    def __post_init__(self):
        object.__setattr__(self, 'observation_ids', frozenset(self.observations.keys()))
        object.__setattr__(self, 'group_ids', frozenset(self.group_data.keys()))
