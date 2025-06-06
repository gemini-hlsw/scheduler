# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from typing import final, Dict, TypeAlias

from lucupy.minimodel import Conditions, Group, NightIndex, TimeslotIndex, UniqueGroupID
import numpy.typing as npt

from .scores import Scores, Metrics


__all__ = [
    'GroupData',
    'GroupDataMap',
    'GroupInfo',
]


@final
@dataclass
class GroupInfo:
    """
    Information regarding Groups that can only be calculated in the Selector.

    Note that the lists here are indexed by night indices as passed to the selection method, or
      equivalently, as defined in the Ranker.

    This comprises:
    1. The most restrictive Conditions required for the group as per all its subgroups.
    2. A flag to indicate if the group can be split.
    3. The nights in which the group can be scheduled based on filtering computed from resource and telescope
       configuration data.
    4. Scoring based on how close the conditions for the group are to the actual conditions.
    5. Scoring based on how the wind affects the group.
    6. A list of indices of the time slots across the nights as to when the group can be scheduled.
    7. The score assigned to the group.
    """
    minimum_conditions: Conditions
    is_splittable: bool
    night_filtering: Dict[NightIndex, bool]
    conditions_score: Dict[NightIndex, npt.NDArray[float]]
    wind_score: Dict[NightIndex, npt.NDArray[float]]
    schedulable_slot_indices: Dict[NightIndex, npt.NDArray[TimeslotIndex]]
    scores: Scores


# Leave this non-immutable as group might change, and we want that change to perpetuate.
@dataclass(frozen=True)
class GroupData:
    """
    Associates Groups with their GroupInfo.
    group is a deep copy.
    """
    # Deep-copied.
    group: Group

    group_info: GroupInfo

    def __iter__(self):
        # Make GroupData unpackable.
        return iter((self.group, self.group_info))


# Map to access GroupData from a UniqueGroupID in the ProgramInfo.
# Since scheduling groups get integer names from OCS, they will overwrite each other if two programs have
# a scheduling group with the same GroupID, so we must make the key include the ProgramID.
GroupDataMap: TypeAlias = Dict[UniqueGroupID, GroupData]
