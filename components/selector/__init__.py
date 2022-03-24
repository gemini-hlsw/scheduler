import logging
import numpy as np
import numpy.typing as npt
import astropy.units as u

from api.observatory.abstract import ObservatoryProperties
from common.minimodel import *
from components.base import SchedulerComponent
from components.collector import Collector, NightIndex
from components.ranker import Ranker, Scores

from typing import Dict, Iterable, FrozenSet, NoReturn, Set


@dataclass
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
    """
    collector: Collector

    # Observatory-specific properties.
    properties: ObservatoryProperties

    def __post_init__(self):
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

    def select(self,
               sites: FrozenSet[Site],
               night_indices: Optional[npt.NDArray[NightIndex]] = None,
               ranker: Optional[Ranker] = None) -> ProgramGroupInfoMap:
        """
        Perform the selection of the groups based on:
        * Resource availability
        * 80% chance of completion (TBD)
        for the given site(s) and night index.

        For each program, begin at the root group and iterate down to the leaves.
        Each leaf contains an Observation. Filter out Observations that cannot be performed
        at one of the given sites.

        For each Observation group node, calculate:
        1. The minimum required conditions to perform the Observation.
        2. For each night index:
           The time slots for which the observation can be performed (based on resource availability and weather).
        3. The score of the Observation.

        Bubble this information back up to conglomerate it for the parent groups.

        An AND group must be able to perform all of its children.

        TODO: Currently OR groups cannot be handled. They are not supported by OCS and
        TODO: attempts to include them will result in a NotImplementedException.
        TODO: Further design work must be done to determine how to score them and how to
        TODO: determine the time slots at which they can be scheduled.

        As all groups are in a group tree rooted at a program node, his information is returned on a
        program-by-program basis.
        """
        # If no night indices are specified, assume all night indices.
        if night_indices is None:
            night_indices = np.arange(len(self.collector.time_grid))

        # If no manual ranker was specified, create the default.
        if ranker is None:
            ranker = Ranker(self.collector, night_indices)

        # The night_indices in the Selector and Ranker must be the same.
        if night_indices != ranker.night_indices:
            raise ValueError(f'The Ranker must have the same night indices as the Selector select method.')

        # Create the structure to hold the mapping fom program ID to its group info.
        program_calcs = {}
        for program_id in Collector.get_program_ids():
            program = Collector.get_program(program_id)
            if program is None:
                logging.error(f'Program {program_id} was not found in the Collector.')
                continue

            # Calculate the group info and put it in the structure if there is actually group
            # info data inside of it, i.e. feasible time slots for it in the plan.
            group_info = self._calculate_group(program.root_group, ranker)

            # TODO: There are many different ways how to structure what we can return.
            # TODO: Right now we return any program that has a schedulable group.
            # TODO: Possible changes to this:
            # TODO: 1. Only return programs where the root group is schedulable.
            # TODO: 2. Only include schedulable groups in the data for each program.
            # Note that the filter on the second will filter our root groups with no slots, since
            # the slots for an ancestor node are based on its subnodes.

            # This will filter out all GroupInfo objects that do not have schedulable slots.
            # group_info = {gid: info for gid, info in group_info.items() if len(info.schedulable_slot_indices) > 0}

            # This filters out any programs that have no root group with any schedulable slots.
            # if len(group_info[program.root_group.id].schedulable_slot_indices) > 0:
            #     program_calcs[program.id] = group_info

            # This filters out any programs that have no groups with any schedulable slots.
            if any(len(info.schedulable_slot_indices) > 0 for info in group_info.values()):
                program_calcs[program.id] = group_info

        # The end product is a map of ProgramID to a map of GroupID to GroupInfo, where
        # at least one GroupInfo has schedulable slots.
        return program_calcs

    def _calculate_group(self,
                         group: Group,
                         ranker: Ranker,
                         group_info_map: GroupInfoMap = None) -> GroupInfoMap:
        """
        Delegate this group to the proper calculation method.
        """
        if group_info_map is None:
            group_info_map = {}

        processor = None
        if isinstance(group, AndGroup):
            if isinstance(group.children, Observation):
                processor = self._calculate_observation_group
            else:
                processor = self._calculate_and_group
        elif isinstance(group, OrGroup):
            processor = self._calculate_or_group

        if processor is None:
            raise ValueError(f'Could not process group {group.id}')

        return processor(group, ranker, group_info_map)

    def _calculate_observation_group(self,
                                     group: Group,
                                     ranker: Ranker,
                                     group_info_map: GroupInfoMap) -> GroupInfoMap:
        """
        Calculate the GroupInfo for a group that contains an observation and add it to
        the group_info_map.
        """
        if not isinstance(group.children, Observation):
            raise ValueError(f'Non-observation group {group.id} cannot be treated as observation group.')

        obs = group.children

        # TODO: Check to make sure that the obs is valid.
        # TODO: If it is not, it should not even be considered for selection.
        if obs.status not in {ObservationStatus.ONGOING, ObservationStatus.READY, ObservationStatus.OBSERVED}:
            return group_info_map

        target_info = Collector.get_target_info(obs.id)

        mrc = obs.constraints.conditions
        is_splittable = len(obs.sequence) > 1

        # Calculate a numpy array of bool indexed by night to determine the resource availability.
        required_res = obs.required_resources()

        # TODO: This is not how we handle standards. Fix this.
        standards = ObservatoryProperties.determine_standard_time(required_res)

        res_night_availability = np.array([self._check_resource_availability(required_res, obs.site, night_idx)
                                           for night_idx in ranker.night_indices])

        if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
            # If we are science or progcal, then the neg HA value for a night is if the first HA for the night
            # is negative.
            negative_hour_angle = np.array([target_info[night_idx].hourangle[0].value < 0
                                            for night_idx in ranker.night_indices])
        else:
            negative_hour_angle = np.array([False] * len(ranker.night_indices))
        too_type = obs.too_type

        # Calculate when the conditions are met and an adjustment array if the conditions are better than needed.
        # TODO: Maybe we only need to concern ourselves with the night indices where the resources are in place.
        conditions_score = []
        wind_score = []
        for night_idx in ranker.night_indices:
            # We need the night_events for the night for timing information.
            night_events = self.collector.get_night_events_for_night_index(obs.site, night_idx)

            # TODO: Is this the time period we want here?
            # TODO: Would this be better with a time grid and times?
            # TODO: We need to decide how we request conditions data.
            time_period = Time(night_events.times[0], night_events.times[-1])
            actual_conditions = self.collector.get_actual_conditions_variant(obs.site, time_period)

            # If we can obtain the conditions variant, calculate the conditions and wind mapping.
            # Otherwise, use arrays of all zeros to indicate that we cannot calculate this information.
            if actual_conditions is not None:
                conditions_score.append(Selector._match_conditions(mrc, actual_conditions, negative_hour_angle, too_type))
                wind_score.append(Selector._wind_conditions(actual_conditions, target_info[night_idx].az))
            else:
                zero = np.zeros(len(night_events.times))
                conditions_score.append(zero.copy())
                wind_score.append(zero.copy())

        # Calculate the schedulable slot indices.
        # These are the indices where the observation has:
        # 1. Visibility
        # 2. Resources available
        # 3. Conditions that are met
        schedulable_slot_indices = []
        for night_idx in ranker.night_indices:
            vis_idx = target_info[night_idx].visibility_slot_idx
            if res_night_availability[night_idx]:
                schedulable_slot_indices[night_idx] = np.where(conditions_score[vis_idx] > 0)[0]
            else:
                schedulable_slot_indices[night_idx] = np.array([])

        # Calculate the scores for the observation across all nights across all timeslots.
        # Multiply by the conditions score to adjust the scores.
        # Note that np.multiply will handle lists of numpy arrays.
        scores = np.multiply(np.multiply(conditions_score, ranker.get_observation_scores(obs.id)), wind_score)

        group_info_map[group.id] = GroupInfo(
            minimum_conditions=mrc,
            is_splittable=is_splittable,
            standards=standards,
            resource_night_availability=res_night_availability,
            conditions_score=conditions_score,
            wind_score=wind_score,
            schedulable_slot_indices=schedulable_slot_indices,
            scores=scores
        )
        return group_info_map

    def _calculate_and_group(self,
                             group: Group,
                             ranker: Ranker,
                             group_info_map: GroupInfoMap) -> GroupInfoMap:
        """
         Calculate the GroupInfo for an AND group that contains subgroups and add it to
        the group_info_map.
        """
        if not isinstance(group, AndGroup):
            raise ValueError(f'Tried to process group {group.id} as an AND group.')
        if isinstance(group.children, Observation):
            raise ValueError(f'Tried to process observation group {group.id} as an AND group.')

        # Process all subgroups and then process this group directly.
        # Ignore the return values here: they will just accumulate in group_info_map.
        for subgroup in group.children:
            self._calculate_group(subgroup, ranker, group_info_map)

        # Make sure that there is an entry for each subgroup. If not, we skip.
        if any(sg.id not in group_info_map for sg in group.children):
            return group_info_map

        # Calculate the most restrictive conditions.
        subgroup_conditions = ([group_info_map[sg.id].minimum_conditions for sg in group.children])
        mrc = Conditions.most_restrictive_conditions(subgroup_conditions)

        # This group will always be splittable unless we have some bizarre nesting.
        is_splittable = len(group.observations()) > 1 or len(group.observations()[0].sequence) > 1

        # TODO: This is not how we handle standards. Fix this.
        standards = np.sum([group_info_map[sg.id].standards for sg in group.children])

        # The availability of resources for this group is the product of resource availability for the subgroups.
        sg_res_night_availability = [group_info_map[sg.id].resource_night_availability for sg in group.children]
        res_night_availability = np.multiply.reduce(sg_res_night_availability).astype(bool)

        # The conditions score is the product of the conditions scores for each subgroup across each night.
        sg_conditions_scores = [group_info_map[sg.id].conditions_score for sg in group.children]
        conditions_score = np.multiply.reduce(sg_conditions_scores)

        # The wind score is the product of the wind scores for each subgroup across each night.
        sg_wind_scores = [group_info_map[sg.id].wind_score for sg in group.children]
        wind_score = np.multiply.reduce(sg_wind_scores)

        # The schedulable slot indices are simply the products of the schedulable slot indices across
        # all the subgroups, which numpy impressively can combine.
        schedulable_slot_indices = np.multiply.reduce([group_info_map[sg.id].schedulable_slot_indices
                                                       for sg in group.children])

        # Calculate the scores for the group across all nights across all timeslots.
        scores = ranker.score_and_group(group)

        group_info_map[group.id] = GroupInfo(
            minimum_conditions=mrc,
            is_splittable=is_splittable,
            standards=standards,
            resource_night_availability=res_night_availability,
            conditions_score=conditions_score,
            wind_score=wind_score,
            schedulable_slot_indices=schedulable_slot_indices,
            scores=scores
        )

        return group_info_map

    def _calculate_or_group(self,
                            group: Group,
                            ranker: Ranker,
                            group_info_map: GroupInfoMap) -> NoReturn:
        """
         Calculate the GroupInfo for an AND group that contains subgroups and add it to
        the group_info_map.

        Not yet implemented.
        """
        raise NotImplementedError(f'Selector does not yet handle OR groups: {group.id}')

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
    def _wind_conditions(variant: Variant,
                         azimuth: Angle) -> npt.NDArray[float]:
        """
        Calculate the effect of the wind conditions on the score of an observation.
        """
        wind = np.ones(len(azimuth))
        az_wd = np.abs(azimuth - variant.wind_dir)
        idx = np.where(np.logical_and(variant.wind_spd > 10 * u.m / u.s,
                                      np.logical_or(az_wd <= variant.wind_sep,
                                                    360. * u.deg - az_wd <= variant.wind_sep)))[0]

        # Adjust down to 0 if the wind conditions are not adequate.
        wind[idx] = 0
        return wind

    @staticmethod
    def _match_conditions(required_conditions: Conditions,
                          actual_conditions: Variant,
                          neg_ha: npt.NDArray[bool],
                          too_status: TooType) -> npt.NDArray[float]:
        """
        Determine if the required conditions are satisfied by the actual conditions variant.
        * required_conditions: the conditions required by an observation
        * actual_conditions: the actual conditions variant, which can hold scalars or numpy arrays
        * neg_ha: a numpy array indexed by night that indicates if the first angle hour is negative
        * too_status: the TOO status of the observation

        We return a numpy array with entries in [0,1] indicating how well the actual conditions match
        the required conditions.

        Note that:
        * 0 indicates that the actual conditions do not satisfy the required ones;
        * 1 indicates that the actual conditions perfectly satisfy the required ones, or we do not care, as is the
            case in certain types of targets of opportunity; and
        * a value in (0,1) indicates that the actual conditions over-satisfy the required ones.
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
        # This should work as we are adjusting structures that are passed by reference.
        def adjuster(array, value):
            better_idx = np.where(np.logical_and(array < value, neg_ha))[0]
            if len(better_idx) > 0 and too_status not in {TooType.RAPID, TooType.INTERRUPT}:
                cmatch[better_idx] = cmatch[better_idx] * array / value
        adjuster(actual_iq, required_conditions.iq)
        adjuster(actual_cc, required_conditions.cc)

        if scalar_input:
            cmatch = np.squeeze(cmatch)
        return cmatch
