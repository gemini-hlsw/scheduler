# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from typing import Dict, List

import numpy.typing as npt
from lucupy.minimodel import Conditions, Group, GroupID, ProgramID

from .scores import Scores


@dataclass(frozen=True)
class GroupInfo:
    """
    Information regarding Groups that can only be calculated in the Selector.

    Note that the lists here are indexed by night indices as passed to the selection method, or
      equivalently, as defined in the Ranker.

    This comprises:
    1. The most restrictive Conditions required for the group as per all its subgroups.
    2. The slots in which the group can be scheduled based on resources and environmental conditions.
    3. The score assigned to the group.
    4. The standards time associated with the group, in hours.
    5. A flag to indicate if the group can be split.
    A group can be split if and only if it contains more than one observation.
    """
    minimum_conditions: Conditions
    is_splittable: bool
    standards: float
    resource_night_availability: npt.NDArray[bool]
    conditions_score: List[npt.NDArray[float]]
    wind_score: List[npt.NDArray[float]]
    schedulable_slot_indices: List[npt.NDArray[int]]
    scores: Scores


@dataclass(frozen=True)
class GroupData:
    """
    Associates Groups with their GroupInfo.
    """
    program_id: ProgramID
    group: Group
    group_info: GroupInfo


# Map to access GroupData from a GroupID in the ProgramInfo.
# Since scheduling groups get integer names from OCS, they will overwrite each other if two programs have
# a scheduling group with the same GroupID, so we must make the key include the ProgramID.
GroupDataMap = Dict[GroupID, GroupData]
