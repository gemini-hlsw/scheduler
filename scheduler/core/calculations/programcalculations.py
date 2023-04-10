# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from typing import Dict

from lucupy.minimodel import NightIndices, UniqueGroupID

from . import GroupData, ProgramInfo


@dataclass(frozen=True)
class ProgramCalculations:
    """
    This dataclass contains a collection of the data returned when a Program is scored.

    The NightIndices are simply the indices over which the Program was scored.

    The group_data_map contains a map from the unique group ID to a GroupData object for all groups that can
    be scheduled during the night indices, i.e. they have a time slot where the score is non-zero.
    If this is empty, there are no schedulable groups.

    The unfiltered_group_data_map contains a list of all group data, including groups that cannot be scheduled.
    This is primarily used for debugging purposes and will be removed for production.

    Another member, has_schedulable_groups, is added after, indicating if the Program has any schedulable groups.
    """
    program_info: ProgramInfo
    night_indices: NightIndices
    group_data_map: Dict[UniqueGroupID, GroupData]
    unfiltered_group_data_map: Dict[UniqueGroupID, GroupData]

    def __post_init__(self):
        """
        Set a convenience variable indicating whether there are any schedulable groups, i.e. group_data_map
        contains entries.
        """
        object.__setattr__('has_schedulable_groups', len(self.group_data_map) > 0)
