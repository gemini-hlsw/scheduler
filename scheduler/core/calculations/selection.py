# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, FrozenSet, Mapping, Optional

from lucupy.helpers import flatten
from lucupy.minimodel import Group, NightIndices, Program, ProgramID, Site, UniqueGroupID

from scheduler.core.components.ranker import Ranker
from scheduler.core.types import StartingTimeslots
from scheduler.core.calculations import GroupData, NightEvents, ProgramCalculations, ProgramInfo


@dataclass(frozen=True)
class Selection:
    """
    The selection of information passed by the Selector to the Optimizer.
    This includes the list of programs that are schedulable and the night event for the nights under consideration.

    Note that the _program_scorer is a configured method to re-score a Program. It carries a lot of data with it,
    and as a result, it is not pickled, so any unpickling of a Selection object will have:
    _program_scorer = None.
    """
    program_info: Mapping[ProgramID, ProgramInfo]
    schedulable_groups: Mapping[UniqueGroupID, GroupData]
    night_events: Mapping[Site, NightEvents]
    night_indices: NightIndices
    time_slot_length: timedelta
    starting_time_slots: StartingTimeslots
    ranker: Ranker

    # Used to re-score programs.
    _program_scorer: Optional[Callable[[Program,
                                        FrozenSet[Site],
                                        NightIndices,
                                        StartingTimeslots,
                                        Ranker],
                              Optional[ProgramCalculations]]] = field(default=None)

    def __reduce__(self):
        """
        Pickle everything but the _program_scorer.
        """
        return (self.__class__, (self.program_info,
                                 self.schedulable_groups,
                                 self.night_indices,
                                 self.night_indices,
                                 self.time_slot_length))

    def score_program(self, program: Program) -> ProgramCalculations:
        """
        Re-score a program. This calls Selector.score_program.

        Note that this will raise a ValueError on unpickled instances of Selection since the
        _program_scorer will be None.
        """
        if self._program_scorer is None:
            raise ValueError('Selection.score_program cannot be called as the selection has a value of None. '
                             'This could happen if the instance was unpickled.')

        return self._program_scorer(program, self.sites, self.night_indices, self.starting_time_slots, self.ranker)

    @property
    def sites(self) -> FrozenSet[Site]:
        return frozenset(self.night_events.keys())

    @staticmethod
    def _get_obs_group_ids(group: Group) -> FrozenSet[UniqueGroupID]:
        """
        Given a group, iterate over the group and return the unique IDs of the observation groups.
        This could be messy due to the groups at different levels, so we flatten in the method that
        uses this group, namely obs_group_ids.

        Example: this might return [1, [2, 3, [4, 5, 6], 7], 8, 9], so we do not specify a return type.
        """
        if group.is_observation_group():
            return frozenset({group.unique_id})
        else:
            return frozenset().union({Selection._get_obs_group_ids(subgroup) for subgroup in group.children})

    def __post_init__(self):
        object.__setattr__(self, 'program_ids', frozenset(self.program_info.keys()))

        # Observation group IDs by frozen set and list.
        obs_group_ids = frozenset(flatten(
            Selection._get_obs_group_ids(program_info.program.root_group) for program_info in self.program_info.values()
        ))
        object.__setattr__(self, 'obs_group_ids', obs_group_ids)
        object.__setattr__(self, 'obs_group_id_list', list(sorted(obs_group_ids)))
