# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from typing import FrozenSet, List, Optional

from lucupy.minimodel import (AndGroup, Atom, Conditions, Constraints, GroupID, Magnitude, NonsiderealTarget,
                              Observation, ObservationClass, OrGroup, Program, ProgramID, QAState, SiderealTarget,
                              Site, Target, TimeAllocation, TimingWindow)


class ProgramProvider(ABC):
    """
    The base class for all populators of the mini-model.

    In all cases, this information will be provided in JSON format, which will
    be a dict (associative array) of key - entry pairs.

    The values of the entries in these pairs will be int, float, or str, so we
    leave them as generic dict.

    Methods can and should only should raise:
    * ValueError if the data is parseable but not of the correct type
    * TypeError if the data is of the wrong type
    * KeyError if a lookup fails in an enum
    * NotImplementedError if the feature is not offered in this provider
    """

    def __init__(self, obs_classes: FrozenSet[ObservationClass]):
        self._obs_classes = obs_classes

    @abstractmethod
    def parse_program(self, data: dict) -> Optional[Program]:
        """
        Given an associative array that contains program data, retrieve the data
        and populate a top-level Program object.

        This method should pass sub-associative arrays to the other methods in
        ProgramProvider to perform the parsing of subelements of Program.

        It is the entry point into the ProgramProvider and should be the only
        method that needs to be directly invoked from outside the class.
        """
        ...

    @abstractmethod
    def parse_or_group(self, data: dict, program_id: ProgramID, group_id: GroupID) -> OrGroup:
        """
        Given an associative array that contains the data needed for an OR group,
        retrieve the data and populate the OrGroup.

        This method should pass sub-associative arrays to the other methods in
        ProgramProvider to perform the parsing of subelements of the OrGroup.

        The members are as follows:
        * group_id: an ID of the group as provided by the data provider

        This method should be called from either parse_program, or, due to group nesting,
        parse_or_group or parse_and_group.
        """
        ...

    @abstractmethod
    def parse_and_group(self, data: dict, program_id: ProgramID, group_id: GroupID) -> Optional[AndGroup]:
        """
        Given an associative array that contains the data needed for an AND group,
        retrieve the data and populate the AndGroup.

        This method should pass sub-associative arrays to the other methods in
        ProgramProvider to perform the parsing of subelements of the OrGroup.

        The members are as follows:
        * group_id: an ID of the group as provided by the data provider

        This method should be called from either parse_program, or, due to group nesting,
        parse_or_group or parse_and_group.
        """
        ...

    @abstractmethod
    def parse_observation(self, data: dict, num: int, program_id: ProgramID) -> Optional[Observation]:
        """
        Given an associative array that contains observation data, retrieve the data
        and populate an Observation object, provided the obs_class for the observation is in
        the

        This method should pass sub-associative arrays to the other methods in
        ProgramProvider to perform the parsing of subelements of the Observation.

        The members are as follows:
        * num: a number associated with the observation as provided by the data provider

        As observations are associated with either programs or AND groups, this method should
        be called from parse_program or parse_and_group.
        """
        ...

    @abstractmethod
    def parse_target(self, data: dict) -> Target:
        """
        Given an associative array that contains common target data, retrieve the general
        data and populate a Target object.

        This method should determine if the target is:
        * sidereal
        * nonsidereal
        and delegate to the appropriate method to complete the process of populating
        the unique data members for SiderealTarget and NonsiderealTarget.

        As targets are associated with observations, this method should be called
        from parse_observation.
        """
        ...

    @abstractmethod
    def parse_sidereal_target(self, data: dict) -> SiderealTarget:
        """
        Given an associative array that contains sidereal target data, retrieve the sidereal
        data and populate a SiderealTarget object.

        This method should only be called from the parse_target method.
        """
        ...

    @abstractmethod
    def parse_nonsidereal_target(self, data: dict) -> NonsiderealTarget:
        """
        Given an associative array that contains nonsidereal target data, retrieve the nonsidereal
        data and populate a NonsiderealTarget object.

        This method should only be called from the parse_target method.
        """
        ...

    @abstractmethod
    def parse_magnitude(self, data: dict) -> Magnitude:
        """
        Given an associative array that contains magnitude data, retrieve the data
        and populate a Magnitude object.
        """
        ...

    @abstractmethod
    def parse_constraints(self, data: dict) -> Constraints:
        """
        Given an associative array that contains constraints data, retrieve the data
        and populate a Constraints object.

        Note that some Observations do not have Constraints associated with them, in which
        case, this method should not be called from parse_observation, which is where it will
        presumably be called.
        """
        ...

    @abstractmethod
    def parse_conditions(self, data: dict) -> Conditions:
        """
        Given an associative array that contains conditions data, retrieve the data and
        populate a Conditions object.

        This should likely be called from the parse_constraints method, but may have other
        applications in handling weather information when it arrives.
        """
        ...

    @abstractmethod
    def parse_timing_window(self, data: dict) -> TimingWindow:
        """
        Given an associative array that contains the data for a single timing window,
        retrieve the data and populate a TimingWindow object.

        As timing windows are associated with observations, this method should be called
        from parse_observation.
        """
        ...

    @abstractmethod
    def parse_time_allocation(self, data: dict) -> TimeAllocation:
        """
        Given an associative array that contains the data for a single time allocation,
        retrieve the data and populate a TimeAllocation object.

        As time allocation information is associated with programs, this method should be
        called from parse_program.
        """
        ...

    @abstractmethod
    def parse_atoms(self, site: Site, sequence: List[dict], qa_states: List[QAState]) -> List[Atom]:
        """
        Given a list of associative arrays from an observation that contain atom data,
        parse / process the atom data and populate a list of Atom objects.

        As atoms are associated with observations, this method should be called from
        parse_observation.
        """
        ...
