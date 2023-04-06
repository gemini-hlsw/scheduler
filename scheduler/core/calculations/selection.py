# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import timedelta
from typing import FrozenSet, Mapping

from lucupy.helpers import flatten
from lucupy.minimodel import Group, ProgramID, Site, UniqueGroupID

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
    schedulable_groups: Mapping[UniqueGroupID, GroupData]
    night_events: Mapping[Site, NightEvents]
    num_nights: int
    time_slot_length: timedelta

    @property
    def sites(self) -> FrozenSet[Site]:
        return frozenset(self.night_events.keys())

    @staticmethod
    def _get_obs_group_ids(group: Group):
        """
        Given a group, iterate over the group and return the unique IDs of the observation groups.
        This could be messy due to the groups at different levels, so we flatten in the method that
        uses this group, namely obs_group_ids.

        Example: this might return [1, [2, 3, [4, 5, 6], 7], 8, 9], so we do not specify a return type.
        """
        if group.is_observation_group():
            return [group.unique_id()]
        else:
            return [Selection._get_obs_group_ids(subgroup) for subgroup in group.children]

    def __post_init__(self):
        object.__setattr__(self, 'program_ids', frozenset(self.program_info.keys()))

        # Observation group IDs by frozen set and list.
        obs_group_ids = frozenset(flatten(
            Selection._get_obs_group_ids(program_info.program.root_group) for program_info in self.program_info.values()
        ))
        object.__setattr__(self, 'obs_group_ids', obs_group_ids)
        object.__setattr__(self, 'obs_group_id_list', list(sorted(obs_group_ids)))
