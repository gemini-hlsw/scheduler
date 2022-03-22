from api.observatory.abstract import ObservatoryProperties
from common.minimodel import *
from components.base import SchedulerComponent
from components.collector import Collector, NightIndex
from components.ranker import Ranker

from typing import Dict, Iterable, FrozenSet, Mapping, Set


@dataclass
class GroupInfo:
    """
    Information regarding Groups that can only be calculated in the Selector.
    This comprises:
    1. The most restrictive Conditions required for the group as per all its subgroups.
    2. The slots in which the group can be scheduled based on resources and environmental conditions.
    3. The score assigned to the group.
    4. The standards time associated with the group, in hours.
    5. A flag to indicate if the group can be split.
    A group can be split if and only if it contains more than one observation.

    TODO: We may have to move the sky brightness calculations in here, although
    TODO: I believe they are based on sun / moon positions so they may not require
    TODO: recalculations.

    TODO: Do we need any more information here?
    """
    minimum_conditions: Conditions
    schedulable_slot_indices: Mapping[(Site, NightIndex), npt.NDArray[int]]
    conditions_score: Mapping[(Site, NightIndex), npt.NDArray[float]]
    score: Mapping[NightIndex, float]
    standards: float
    is_splittable: bool

# @dataclass
# class Visit:
#     idx: int
#     site: Site
#     group: Group
#
#     # Standards time in time slots.
#     standard_time: int
#
#     def __post_init__(self):
#         self.score = None
#
#         # Create a Conditions object that uses the most restrictive values over all
#         # observations.
#         self.observations = self.group.observations()
#
#         # TODO: Test this.
#         self.conditions = Conditions(
#             cc=CloudCover(min(flatten((obs.constraints.conditions.cc for obs in self.observations)))),
#             iq=ImageQuality(min(flatten((obs.constraints.conditions.iq for obs in self.observations)))),
#             sb=SkyBackground(min(flatten((obs.constraints.conditions.sb for obs in self.observations)))),
#             wv=WaterVapor(min(flatten((obs.constraints.conditions.wv for obs in self.observations))))
#         )
#
#     def length(self) -> int:
#         """
#         Calculate the length of the unit based on both observation and calibrations times
#         """
#         # length here is acq + time and is a float?
#         obs_slots = sum([obs.length for obs in self.observations])
#
#         if self.standard_time > 0:  # not a single science observation
#             standards_needed = max(1, int(obs_slots // self.standard_time))
#
#             if standards_needed == 1:
#                 cal_slots = self.calibrations[0].length  # take just one
#             else:
#                 cal_slots = sum([cal.length for cal in self.calibrations])
#
#             return obs_slots + cal_slots
#         else:
#             return obs_slots
#
#     def observed(self) -> int:
#         """
#         Calculate the observed time for both observation and calibrations
#         """
#         # observed: observation time on time slots
#         obs_slots = sum([obs.observed for obs in self.observations])
#         cal_slots = sum([cal.observed for cal in self.calibrations])
#         return obs_slots + cal_slots
#
#     def acquisition(self) -> None:
#         """
#         Add acquisition overhead to the total length of each observation in the unit
#         """
#         for observation in self.observations:
#             if observation.observed < observation.length:  # not complete observation
#                 observation.length += observation.acquisition
#
#     def get_observations(self) -> Dict[int, Observation]:
#         total_obs = {}
#         for obs in self.observations:
#             total_obs[obs.idx] = obs
#         for cal in self.calibrations:
#             total_obs[cal.idx] = cal
#         return total_obs
#
#     def airmass(self, obs_idx: int) -> float:
#         """
#         Get airmass values for the observation
#         """
#         if obs_idx in self.observations:
#             return self.observations[obs_idx].visibility.airmass
#         if obs_idx in self.calibrations:
#             return self.calibrations[obs_idx].visibility.airmass
#         else:
#             return None
#
#     def __contains__(self, obs_idx: int) -> bool:
#
#         if obs_idx in [sci.idx for sci in self.observations]:
#             return True
#         elif obs_idx in [cal.idx for cal in self.calibrations]:
#             return True
#         else:
#             return False
#
#     def __str__(self) -> str:
#         return f'Visit {self.idx} \n ' + \
#                f'-- observations: \n {[str(obs) for obs in self.observations]} \n' + \
#                f'-- calibrations: {[str(cal) for cal in self.calibrations]} \n'


# Type alias for Group information mapping.
GroupInfoMap = Dict[GroupID, GroupInfo]
ProgramGroupInfoMap = Dict[ProgramID, GroupInfoMap]


@dataclass
class Selector(SchedulerComponent):
    """
    This is the Selector portion of the automated Scheduler.
    It selects the scheduling candidates that are viable for the data collected by
    the Collector.

    Note that unlike the Collector, the Selector does not use static variables, since
    the data contained here can change over time, unlike the Collector where the
    information is statically determined.

    TODO: Is this what we want?
    """
    collector: Collector

    # Observatory-specific properties.
    properties: ObservatoryProperties

    # Use the default Ranker to calculate scores.
    ranker: Ranker = Ranker()

    def __init__(self):
        """
        Initialize internal non-input data members.
        """
        # Visibility calculations that can only be completed in the Selector.
        # These comprise what can be done on a given night depending on weather,
        # resources, etc.
        self._groups: Dict[GroupID, Group] = {}

        # List of groups that have been selected.
        self._group_info: GroupInfoMap = {}

    def get_group_ids(self) -> Iterable[GroupID]:
        """
        Return a list of all the group IDs stored in the Selector.
        """
        return self._groups.keys()

    def get_group(self, group_id: GroupID) -> Optional[Group]:
        """
        If a group with the given ID exists, return it.
        Otherwise, return None.
        """
        return self._groups.get(group_id, None)

    def get_group_info(self, group_id) -> Optional[GroupInfoMap]:
        """
        Given a GroupID, if the group exists, return the group information
        as a map from NightIndex to GroupInfo.
        """
        return self._group_info.get(group_id, None)

    def _calculate_group(self,
                         group: Group,
                         sites: FrozenSet[Site],
                         night_indices: npt.NDArray[NightIndex],
                         group_info_map: GroupInfoMap) -> (GroupInfoMap, Conditions):
        """
        Calculate the information for a Group as described in select.
        We ignore the return result of GroupInfo in all cases except the root
        group, which gets returned automatically.
        The Conditions are the minimum required conditions for all subgroups.
        """
        # Negative hour angle: an array of boolean over the night indices.
        # Assume True, but if any observation in the group has an obsclass of SCIENCE or
        # PROGRAMCAL, and the visibility of the observation for the night has an hour angle[0]
        # value > 0, then set it to False for the night.
        if isinstance(group.children, Observation):
            # Process this group directly.
            obs = group.children
            mrc = obs.constraints.conditions
            is_splittable = False

            # For a given night, if the hour angle is < 0 in the first time step, then we don't consider it
            # setting at the start.
            visibility = Collector.get_target_info(obs.id)

            # We only worry about negative hour angle if the observation is SCIENCE or PROGCAL.
            if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
                # If we are science or progcal, then the neg HA value for a night is if the first HA for the night
                # is negative.
                negative_hour_angle = {night_idx: visibility[night_idx].hourangle[0].value < 0
                                       for night_idx in night_indices}
            else:
                negative_hour_angle = {night_idx: False for night_idx in night_indices}

        else:
            # TODO: Design of handling of OR subgroups.
            # First off, we ensure that this group does not have any OR subgroups.
            # We do not know how to handle an OR subgroup at this phase, so OR groups
            # are only allowed as the root group.
            if any(isinstance(subgroup, OrGroup) for subgroup in group.children):
                raise NotImplementedError(f'Group ${group.id} has an OrGroup as a subgroup.')

            # Process all subgroups and then process this group directly.
            # First, we get the Conditions for all subgroups and find the most restrictive,
            # which is what we need if we want to satisfy the group requirements.
            conditions = {}
            for subgroup in group.children:
                conditions = {cd for _, cd in self._calculate_group(subgroup, sites, night_indices, group_info_map)}
            mrc = Conditions.most_restrictive_conditions(conditions.union(Constraints.LEAST_RESTRICTIVE_CONDITIONS))

            # This will likely always be true unless we have trivial nested groups, but check
            # to make sure.
            is_splittable = len(group.observations()) > 1

        # TODO: We should probably store or return the required resources instead of
        # TODO: recalculating them repeatedly since this calculation is inefficient
        # TODO: and requires re-traversing the entire tree rooted at this group.
        # Calculate the nights where the required resources are in place.
        resources = group.required_resources()
        res_night_index = {(site, night_index) for site in sites for night_index in night_indices
                           if Selector._check_resource_availability(resources, site, night_index)}

        # Calculate when the conditions are met.
        # We only need to concern ourselves with the night indices where the resources are in place.
        # We can skip anywhere the resources are not available.
        conditions_score = {}
        for (site, night_index) in res_night_index:
            # TODO: This doesn't look quite right.
            night_events = self.collector.get_night_events_for_night_index(site, night_index)

            # TODO: Is this the time period we want here?
            # TODO: Would this be better with a time grid and times?
            # TODO: We need to decide how we request conditions data.
            time_period = Time(night_events.times[0], night_events.times[-1])
            actual_conditions = self.collector.get_actual_conditions_variant(site, time_period)

            if actual_conditions is not None:
                # TODO: *** Figure out the negha and ToOType. ***
                conditions_score[(site, night_index)] = self._match_conditions(mrc, actual_conditions,
                                                                               False, TooType.RAPID)
                # TODO: Filter out the visibility_slot_idx based on the conditions score.
                # TODO: This is on observations. How do we combine the observations?
                target_info_night_idx_map = self.collector.get_target_info()

        group_info_map[group.id] = GroupInfo(
            minimum_conditions=mrc,
            schedulable_slot_indices=None,
            conditions_score=conditions_score,
            score=None,
            standards=0,
            is_splittable=is_splittable
        )

        return group_info_map, mrc

    def select(self,
               sites: FrozenSet[Site],
               night_indices: FrozenSet[NightIndex]) -> ProgramGroupInfoMap:
        """
        Perform the selection of the observations and groups based on:
        * Resource availability
        * 80% chance of completion (TBD)
        for the given site(s) and night index.

        For each program, begin at the root group and iterate down to the leaves.
        Each leaf contains an Observation. Filter out Observations that cannot be performed
        at one of the given sites.

        For each Observation group node, calculate:
        1. The minimum required conditions to perform the Observation.
        1. For each night index:
           The time slots for which the observation can be performed (based on resource availability and weather).
        3. The score of the Observation.

        Bubble this information back up to conglomerate it for the parent groups.

        Note that this requires special handling for OR Groups: an OR group may still
        be performed if some of its children cannot be performed.

        An AND group must be able to perform all of its children.

        This information is returned on a program-by-program basis.
        """
        return {program_id: self.calculate_group(Collector.get_program(program_id).root_group, sites, night_indices)
                for program_id in Collector.get_program_ids()}

    @staticmethod
    def _check_resource_availability(required_resources: Set[Resource],
                                     site: Site,
                                     night_idx: NightIndex) -> bool:
        """
        Determine if the required resources as listed are available at
        the specified site during the given time_period, and if so, at what
        intervals in the time period.
        """
        available_resources = Collector.available_resources(site, night_idx)
        return required_resources.issubset(available_resources)

    @staticmethod
    def _match_conditions(required_conditions: Conditions,
                          actual_conditions: Variant,
                          neg_ha: bool,
                          too_status: TooType) -> npt.NDArray[float]:
        """
        Determine if the required conditions are satisfied by the actual conditions variant.
        """
        # TODO: Can we move part of this to the mini-model? Do we want to?

        # Convert the actual conditions to arrays.
        actual_iq = np.asarray(actual_conditions.iq.value)
        actual_cc = np.asarray(actual_conditions.cc.value)
        actual_wv = np.asarray(actual_conditions.wv.value)

        # The mini-model manages the IQ, CC, WV, and SB being the same types and sizes
        # so this check is a bit extraneous: if one has an ndim of 0, they all will.
        scalar_input = actual_iq.ndim == 0 or actual_cc.ndim == 0 or actual_wv.ndim == 0
        if actual_iq.ndim == 0:
            actual_iq = actual_iq[None]
        if actual_cc.ndim == 0:
            actual_cc = actual_cc[None]
        if actual_wv.ndim == 0:
            actual_wv = actual_wv[None]

        # Again, all lengths are guaranteed to be the same by the mini-model.
        length = len(actual_iq)

        # We must be working with required_conditions that are either scalars or the same
        # length as the actual_conditions.
        if len(required_conditions) not in {1, length}:
            raise ValueError(f'Cannot match conditions between arrays of different sizes: {actual_conditions} '
                             f'and {required_conditions}')

        # Determine the positions where the actual conditions are worse than the requirements.
        bad_iq = actual_iq > required_conditions.iq
        bad_cc = actual_cc > required_conditions.cc
        bad_wv = actual_wv > required_conditions.wv

        bad_cond_idx = np.where(np.logical_or(np.logical_or(bad_iq, bad_cc), bad_wv))[0]
        cmatch = np.ones(length)
        cmatch[bad_cond_idx] = 0

        # Penalize for using IQ / CC that is better than needed:
        # Multiply the weights by actual value / value where value is better than required and target
        # does not set soon and is not a rapid ToO.
        # TODO: What about interrupting ToOs?

        # This should work as we are adjusting structures that are passed by reference.
        def adjuster(array, value):
            better_idx = np.where(array < value)[0]
            if len(better_idx) > 0 and neg_ha and too_status != TooType.RAPID:
                cmatch[better_idx] = cmatch[better_idx] * array / value
        adjuster(actual_iq, required_conditions.iq)
        adjuster(actual_cc, required_conditions.cc)

        if scalar_input:
            cmatch = np.squeeze(cmatch)
        return cmatch
