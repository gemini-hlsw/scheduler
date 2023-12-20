# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from typing import FrozenSet, Mapping

from lucupy.minimodel import GroupID, Program, NightIndices, Observation, ObservationID, ROOT_GROUP_ID, UniqueGroupID

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
    group_data_map: GroupDataMap

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
        object.__setattr__(self, 'group_ids', frozenset(self.group_data_map.keys()))


@dataclass(frozen=True)
class ProgramCalculations:
    """
    This dataclass contains a collection of the data returned when a Program is scored.

    The NightIndices are simply the indices over which the Program was scored.

    The group_data_map contains a map from the unique group ID to a GroupData object for all groups that can
    be scheduled during the night indices, i.e. they have a time slot where the score is non-zero.
    If this is empty, there are no schedulable groups.

    The top_level_group_data_map contains only the top-level groups in group_data_map.

    The unfiltered_group_data_map contains a list of all group data, including groups that cannot be scheduled.
    This is primarily used for debugging purposes and will be removed for production.

    Members set post-init:
    * has_schedulable_groups, indicating if the Program has any schedulable groups.
    * top_level_groups, a FrozenSet of UniqueGroupID for the top-level groups.
    """
    program_info: ProgramInfo
    night_indices: NightIndices
    group_data_map: GroupDataMap
    unfiltered_group_data_map: GroupDataMap

    has_schedulable_groups: bool = field(init=False)
    top_level_groups: FrozenSet[UniqueGroupID] = field(init=False)

    def __post_init__(self):
        """
        Set a convenience variable indicating whether there are any schedulable groups, i.e. group_data_map
        contains entries.
        """
        object.__setattr__(self, 'has_schedulable_groups', len(self.group_data_map) > 0)

        # Get the top level groups, i.e. the groups such that their parent group is not in the group_data_map.
        # Get all scheduling groups excluding the root.
        # We compare on GroupID here instead of UniqueGroupID since it is easier as all root groups have the same
        # GroupID, but different UniqueGroupIDs.
        scheduling_groups = [group for (group, _) in self.group_data_map.values()
                             if group.id != ROOT_GROUP_ID and group.is_scheduling_group()]

        # Extract the children's names of all scheduling groups except the root into a set.
        # This way, we only have to check IDs for equality instead of the entire group structure.
        scheduling_group_children_names = {
            child.unique_id for group in scheduling_groups for child in group.children
        }

        # Find the group_data for groups that are not the root group and are not in any scheduling_group.
        top_level_groups = frozenset({group_data.group.unique_id for group_data in self.group_data_map.values()
                                      if (group_data.group.id != ROOT_GROUP_ID and
                                          group_data.group.unique_id not in scheduling_group_children_names)})

        object.__setattr__(self, 'top_level_groups', top_level_groups)
