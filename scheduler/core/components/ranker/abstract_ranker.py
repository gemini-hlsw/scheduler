# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod, ABC
from typing import Dict, FrozenSet

import numpy as np
import numpy.typing as npt
from lucupy.minimodel import ALL_SITES, AndGroup, OrGroup, Group, NightIndex, NightIndices, Observation, Program, Site

from scheduler.core.calculations import Scores, GroupDataMap


__all__ = [
    'AbstractRanker',
]


class AbstractRanker(ABC):
    def __init__(self,
                 collector: 'Collector',
                 night_indices: NightIndices,
                 sites: FrozenSet[Site] = ALL_SITES):
        """
        We only want to calculate the parameters once since they do not change.
        """
        self.collector = collector
        self.night_indices = night_indices
        self.sites = sites

        # For convenience, for each site, create:
        # 1. An empty observation score array.
        # 2. An empty group scoring array, used to collect the group scores.
        # This allows us to avoid having to store a reference to the Collector in the Ranker.
        self._empty_obs_scores: Dict[Site, Dict[NightIndex, npt.NDArray[float]]] = {}
        self._empty_group_scores: Dict[Site, Dict[NightIndex, npt.NDArray[float]]] = {}
        for site in self.sites:
            night_events = collector.get_night_events(site)

            # Create a full zero score that fits the sites, nights, and time slots for observations.
            self._empty_obs_scores[site] = {night_idx: np.zeros(len(night_events.times[night_idx]), dtype=float)
                                            for night_idx in self.night_indices}

            # Create a full zero score that fits the sites, nights, and time slots for group calculations.
            # As this must collect the subgroups, the dimensions are different from observation scores.
            self._empty_group_scores[site] = {night_idx: np.zeros((0, len(night_events.times[night_idx])), dtype=float)
                                              for night_idx in self.night_indices}

    def score_group(self, group: Group, group_data_map: GroupDataMap) -> Scores:
        """
        Calculate the score of a Group.

        This method returns the results in the form of a list, where each entry represents
        one night as per the night_indices array, with the list entries being numpy arrays
        that contain the scoring for each time slot across the night.
        """
        # Check isinstance instead of is_and_group or is_or_group because otherwise, we get warnings.
        if isinstance(group, AndGroup):
            return self._score_and_group(group, group_data_map)
        elif isinstance(group, OrGroup):
            return self._score_or_group(group, group_data_map)
        else:
            raise ValueError('Ranker group scoring can only score groups.')

    @abstractmethod
    def score_observation(self, program: Program, obs: Observation) -> Scores:
        """
        Calculate the scores for an observation for each night for each time slot index.
        These are returned as a list indexed by night index as per the night_indices supplied,
        and the list items are numpy arrays of float for each time slot during the specified night.
        """

    @abstractmethod
    def _score_and_group(self, group: AndGroup, group_data_map: GroupDataMap) -> Scores:
        """
        Calculate the scores for each night and time slot of an AND Group.
        """

    @abstractmethod
    def _score_or_group(self, group: OrGroup, group_data_map: GroupDataMap) -> Scores:
        """
        Calculate the scores for each night and time slot of an OR Group.
        TODO: This is TBD and requires more design work.
        TODO: In fact, OcsProgramProvider does not even support OR Groups.
        """
