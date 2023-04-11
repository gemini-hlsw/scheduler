# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod
from typing import Dict, FrozenSet, List

import numpy as np
import numpy.typing as npt
from lucupy.minimodel import ALL_SITES, AndGroup, OrGroup, Group, NightIndices, Observation, Program, Site

from scheduler.core.calculations import Scores, GroupDataMap
from scheduler.core.components.collector import Collector


class Ranker:
    def __init__(self,
                 collector: Collector,
                 night_indices: NightIndices,
                 sites: FrozenSet[Site] = ALL_SITES):
        """
        We only want to calculate the parameters once since they do not change.
        """
        self.collector = collector
        self.night_indices = night_indices
        self.sites = sites

        # Create a full zero score that fits the sites, nights, and time slots for initialization
        # and to return if an observation is not to be included.
        self._zero_scores: Dict[Site, List[npt.NDArray[float]]] = {}
        for site in self.collector.sites:
            night_events = self.collector.get_night_events(site)
            self._zero_scores[site] = [np.zeros(len(night_events.times[night_idx])) for night_idx in self.night_indices]

    def score_group(self, group: Group, group_data_map: GroupDataMap) -> Scores:
        """
        Calculate the score of a Group.
        This is reliant on all the Observations in the Group being scored, which
        should be automatically done when the Ranker is created and given a Collector.

        This method returns the results in the form of a list, where each entry represents
        one night as per the night_indices array, with the list entries being numpy arrays
        that contain the scoring for each time slot across the night.
        """
        # Determine which type of Group we are working with.
        # We check isinstance instead of is_and_group or is_or_group because otherwise, we get warnings.
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
