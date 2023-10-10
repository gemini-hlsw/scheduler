# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, FrozenSet, Optional, final

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import Angle
from astropy.units import Quantity
from lucupy.helpers import is_contiguous
from lucupy.minimodel import (AndGroup, Conditions, Group, Observation, ObservationClass, ObservationStatus, Program,
                              ProgramID, ROOT_GROUP_ID, Site, TooType, NightIndex, NightIndices, UniqueGroupID, Variant)
from lucupy.minimodel import CloudCover, ImageQuality

from scheduler.core.calculations import GroupData, GroupDataMap, GroupInfo, ProgramCalculations, ProgramInfo, Selection
from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.collector import Collector
from scheduler.core.components.ranker import DefaultRanker, Ranker
from scheduler.core.types import StartingTimeslots
from scheduler.services import logger_factory
from scheduler.services.resource import NightConfiguration

logger = logger_factory.create_logger(__name__)

# Aliases to pass around resource availability information for sites and night indices.
NightConfigurations = Dict[NightIndex, NightConfiguration]
NightConfigurationData = Dict[Site, NightConfigurations]


@final
@dataclass(frozen=True)
class Selector(SchedulerComponent):
    """
    This is the Selector portion of the automated Scheduler.
    It selects the scheduling candidates that are viable for the data collected by
    the Collector.

    The collector is the data repository that contains all the data and calculations necessary for scheduling.
    The num_nights indicates the number of nights for which we wish to schedule.
    """
    collector: Collector
    num_nights_to_schedule: int
    default_cc: CloudCover = CloudCover.CC50
    default_iq: ImageQuality = ImageQuality.IQ70

    _wind_sep: ClassVar[Angle] = 20. * u.deg
    _wind_spd_bound: ClassVar[Quantity] = 10. * u.m / u.s

    def __post_init__(self):
        if (self.num_nights_to_schedule < 0 or
                self.num_nights_to_schedule > self.collector.num_nights_calculated):
            raise ValueError(f'Scheduling requested for {self.num_nights_to_schedule} nights, but visibility '
                             f'calculations only performed for {self.collector.num_nights_calculated}. '
                             'Cannot proceed.')

    @staticmethod
    def _process_starting_time_slots(sites: FrozenSet[Site],
                                     night_indices: NightIndices,
                                     starting_time_slots: Optional[StartingTimeslots]) -> StartingTimeslots:
        if starting_time_slots is None:
            starting_time_slots = {}

        # Make sure we have an entry for all relevant sites for all night indices.
        # Add starting index of 0 if there is no data.
        for site in sites:
            night_dict = starting_time_slots.setdefault(site, {})
            night_dict.update({night_idx: 0 for night_idx in night_indices if night_idx not in night_dict})

            # Check for extra keys.
            if extra_keys := night_dict.keys() - night_indices:
                raise ValueError(f'Extra night indices for site {site.name}: {extra_keys}')

        return starting_time_slots

    def select(self,
               sites: Optional[FrozenSet[Site]] = None,
               night_indices: Optional[NightIndices] = None,
               starting_time_slots: Optional[StartingTimeslots] = None,
               ranker: Optional[Ranker] = None) -> Selection:
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
        """
        if sites is None:
            sites = self.collector.sites
        if not sites:
            raise ValueError('Attempted to fetch a selection over no sites.')

        # NOTE: If night_indices is None, assume the whole calculation period.
        if night_indices is None:
            night_indices = np.arange(len(self.collector.time_grid))
        if not is_contiguous(night_indices):
            raise ValueError(f'Attempted to select a non-contiguous set of night indices: {set(night_indices)}')
        night_indices = np.array(sorted(night_indices))

        # Set the starting time slots dictionary as necessary.
        starting_time_slots = Selector._process_starting_time_slots(sites, night_indices, starting_time_slots)

        # If no manual ranker was specified, create the default.
        if ranker is None:
            ranker = DefaultRanker(self.collector, night_indices, sites)

        # Create the structure to hold the mapping fom program ID to its group info.
        program_info_map: Dict[ProgramID, ProgramInfo] = {}

        # A flat top-level list of GroupData indexed by UniqueGroupID.
        schedulable_groups_map: Dict[UniqueGroupID, GroupData] = {}

        for program_id in Collector.get_program_ids():
            original_program = Collector.get_program(program_id)
            if original_program is None:
                logger.error(f'Program {program_id} was not found in the Collector.')
                continue

            # We make a deep copy of the Program to work with to not change the Program in the Collector.
            # This will allow us to use the members of this deep copy for things like internal time accounting
            # while leaving the information in the Collector intact.
            program = deepcopy(original_program)
            program_calculations = self.score_program(program, sites, night_indices, starting_time_slots, ranker)
            if program_calculations is None:
                # Warning is already issued in scorer.
                continue

            # Get the top-level groups (excluding root) in group_data_map and add to the schedulable_groups_map map.
            for unique_group_id in program_calculations.top_level_groups:
                group_data = program_calculations.group_data_map[unique_group_id]
                schedulable_groups_map[group_data.group.unique_id] = group_data

            program_info_map[program.id] = program_calculations.program_info

        # The end product is a map of ProgramID to a map of GroupID to GroupInfo, where
        return Selection(
            program_info=program_info_map,
            schedulable_groups=schedulable_groups_map,
            night_events={site: self.collector.get_night_events(site) for site in sites},
            night_indices=night_indices,
            starting_time_slots=starting_time_slots,
            time_slot_length=self.collector.time_slot_length.to_datetime(),
            ranker=ranker,
            _program_scorer=self.score_program
        )
        # at least one GroupInfo has schedulable slots.

    def score_program(self,
                      program: Program,
                      sites: FrozenSet[Site],
                      night_indices: NightIndices,
                      starting_time_slots: StartingTimeslots,
                      ranker: Ranker) -> Optional[ProgramCalculations]:
        """
        Given a program and an array of night indices, score the program for the specified night indices
        starting at the specified time slot.

        The program's score prior to the starting time slot for a night is set to 0.
        This is to allow for scoring of partial nights.

        The Ranker can be specified. In the case that it is not, the DefaultRanker is used.

        If the sites used by the Program do not intersect the sites parameter, then None is returned.
        Otherwise, the data is bundled in a ProgramCalculations object.
        """
        # The night_indices in the Selector must be a subset of the Ranker.
        night_indices_set = set(night_indices)
        if not night_indices_set.issubset(ranker.night_indices):
            ranker_night_indices = set(ranker.night_indices)
            invalid_night_indices = night_indices_set - ranker_night_indices
            raise ValueError(f'Selector is attempting to score program {program.id} on invalid night indices:\n'
                             f'\tAvailable night indices: {ranker.night_indices}\n'
                             f'\tNight indices to score: {night_indices_set}\n'
                             f'\tInvalid night indices: {invalid_night_indices}')

        # Get the night configuration for all nights.
        night_configurations = {site: self.collector.night_configurations(site, night_indices) for site in sites}

        # TODO: We have to check across nights.
        # Calculate the group info and put it in the structure if there is actually group
        # info data inside it, i.e. feasible time slots for it in the plan.
        # This will filter out all GroupInfo objects that do not have schedulable slots.
        unfiltered_group_data_map = self._calculate_group(program,
                                                          program.root_group,
                                                          sites,
                                                          night_indices,
                                                          starting_time_slots,
                                                          night_configurations,
                                                          ranker)

        # We want to check if there are any time slots where a group can be scheduled: otherwise, we omit it.
        group_data_map = {gp_id: gp_data for gp_id, gp_data in unfiltered_group_data_map.items()
                          if any(len(indices) > 0 for indices in gp_data.group_info.schedulable_slot_indices.values())}

        # In an observation group, the only child is an Observation:
        # hence, references here to group.children are simply the Observation.
        observations = {group_data.group.children.id: group_data.group.children
                        for group_data in group_data_map.values()
                        if group_data.group.is_observation_group()}
        target_info = {obs_id: self.collector.get_target_info(obs_id) for obs_id, obs in observations.items()}

        program_info = ProgramInfo(
            program=program,
            group_data_map=group_data_map,
            observations=observations,
            target_info=target_info
        )

        return ProgramCalculations(
            program_info=program_info,
            night_indices=night_indices,
            group_data_map=group_data_map,
            unfiltered_group_data_map=unfiltered_group_data_map
        )

    def _calculate_group(self,
                         program: Program,
                         group: Group,
                         sites: FrozenSet[Site],
                         night_indices: NightIndices,
                         starting_time_slots: StartingTimeslots,
                         night_configurations: NightConfigurationData,
                         ranker: Ranker,
                         group_data_map: GroupDataMap = None) -> GroupDataMap:
        """
        Delegate this group to the proper calculation method.
        """
        if group_data_map is None:
            group_data_map: GroupDataMap = {}

        if group.is_observation_group():
            processor = self._calculate_observation_group
        elif group.is_and_group():
            processor = self._calculate_and_group
        elif group.is_or_group():
            processor = self._calculate_or_group
        else:
            raise ValueError(f'Could not process group {group.id}')

        return processor(program,
                         group,
                         sites,
                         night_indices,
                         starting_time_slots,
                         night_configurations,
                         ranker,
                         group_data_map)

    def _calculate_observation_group(self,
                                     program: Program,
                                     group: Group,
                                     sites: FrozenSet[Site],
                                     night_indices: NightIndices,
                                     starting_time_slots: StartingTimeslots,
                                     night_configurations: NightConfigurationData,
                                     ranker: Ranker,
                                     group_data_map: GroupDataMap) -> GroupDataMap:
        """
        Calculate the GroupInfo for a group that contains an observation and add it to
        the group_data_map.

        Note that the scores for times before the starting time slot for a given night index will be set to zero.
        All other components of the score that are stored will be maintained.

        TODO: Not every observation has TargetInfo: if it is not in a specified class, it will not.
        TODO: How do we handle these cases? For now, I am going to skip any observation with a missing TargetInfo.
        """
        if not group.is_observation_group():
            raise ValueError(f'Non-observation group {group.id} cannot be treated as observation group.')

        obs = group.children

        # Only schedule ONGOING or READY observations.
        if obs.status not in {ObservationStatus.ONGOING, ObservationStatus.READY}:
            logger.warning(f'Selector skipping observation {obs.id}: status is {obs.status.name}.')
            return group_data_map

        # This should never happen.
        if obs.site not in sites:
            logger.error(f'Selector requested to score {obs.id}: not at a designated site.')
            return group_data_map

        # We ignore the Observation if:
        # 1. There is no target info associated with it.
        target_info = Collector.get_target_info(obs.id)
        if target_info is None:
            logger.warning(f'Selector skipping observation {obs.id}: no target info.')
            return group_data_map
        # 2. There are no constraints associated with it.
        if obs.constraints is None:
            logger.warning(f'Selector skipping observation {obs.id}: no conditions.')
            return group_data_map

        mrc = obs.constraints.conditions
        is_splittable = len(obs.sequence) > 1

        # TODO: Do we need standards?
        # TODO: By the old Visit code, we had them and handled as per comment.
        # TODO: How do we handle them now?
        # standards = ObservatoryProperties.determine_standard_time(required_res, obs.wavelengths(), obs)
        standards = 0.

        # Calculate a numpy array of bool indexed by night to determine when the group can be added to the plan
        # based on the night configuration filtering.
        night_filtering = {night_idx: night_configurations[obs.site][night_idx].filter.group_filter(group)
                           for night_idx in night_indices}

        if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
            # If we are science or progcal, then the neg HA value for a night is if the first HA for the night
            # is negative.
            neg_ha = {night_idx: target_info[night_idx].hourangle[0].value < 0 for night_idx in night_indices}
        else:
            neg_ha = {night_idx: False for night_idx in night_indices}
        too_type = obs.too_type

        # Calculate when the conditions are met and an adjustment array if the conditions are better than needed.
        # TODO: Maybe we only need to concern ourselves with the night indices where the resources are in place.
        conditions_score = {}
        wind_score = {}

        # We need the night_events for the night for timing information.
        night_events = self.collector.get_night_events(obs.site)

        for night_idx in night_indices:
            # Get the conditions for the night.
            start_time = night_events.times[night_idx][0]
            end_time = night_events.times[night_idx][-1]
            actual_conditions = self.collector.sources.origin.env.get_actual_conditions_variant(obs.site,
                                                                                                start_time,
                                                                                                end_time)

            variant_length = len(actual_conditions.cc)
            actual_conditions = Variant(iq=np.array([self.default_iq] * variant_length),
                                        cc=np.array([self.default_cc] * variant_length),
                                        wind_dir=actual_conditions.wind_dir,
                                        wind_spd=actual_conditions.wind_spd)

            # Make sure that we have conditions for every time slot.
            num_time_slots = len(night_events.times[night_idx])
            if variant_length != num_time_slots:
                error_str = (f'Night {night_idx} has {num_time_slots} entries, '
                             f'but only {variant_length} conditions points.')
                logger.error(error_str)
                raise ValueError(error_str)

            # If we can obtain the conditions variant, calculate the conditions and wind mapping.
            # Otherwise, use arrays of all zeros to indicate that we cannot calculate this information.
            if actual_conditions is not None:
                conditions_score[night_idx] = Selector._match_conditions(mrc,
                                                                         actual_conditions,
                                                                         neg_ha[night_idx],
                                                                         too_type)
                wind_score[night_idx] = Selector._wind_conditions(actual_conditions, target_info[night_idx].az)
            else:
                zero = np.zeros(len(night_events.times[night_idx]))
                conditions_score[night_idx] = zero.copy()
                wind_score[night_idx] = zero.copy()

        # Calculate the schedulable slot indices.
        # These are the indices where the observation has:
        # 1. Visibility
        # 2. Resources available
        # 3. Conditions that are met
        schedulable_slot_indices = {}
        for night_idx in night_indices:
            vis_idx = target_info[night_idx].visibility_slot_idx
            if night_filtering[night_idx]:
                schedulable_slot_indices[night_idx] = np.where(conditions_score[night_idx][vis_idx] > 0)[0]
            else:
                schedulable_slot_indices[night_idx] = np.array([])

        obs_scores = ranker.score_observation(program, obs)

        # Calculate the scores for the observation across all night indices across all timeslots.
        scores = {night_idx: np.multiply(
                np.multiply(conditions_score[night_idx], obs_scores[night_idx]),
                wind_score[night_idx]) for night_idx in night_indices}

        # Zero out the data for each night index's starting time slots prior to the value specified.
        starting_time_slots_for_site = starting_time_slots[obs.site]
        for night_idx, time_slot_idx in starting_time_slots_for_site.items():
            scores[night_idx][:time_slot_idx] = 0.0

        # These scores might differ from the observation score in the ranker since they have been adjusted for
        # conditions and wind.
        group_info = GroupInfo(
            minimum_conditions=mrc,
            is_splittable=is_splittable,
            standards=standards,
            night_filtering=night_filtering,
            conditions_score=conditions_score,
            wind_score=wind_score,
            schedulable_slot_indices=schedulable_slot_indices,
            scores=scores
        )

        group_data_map[group.unique_id] = GroupData(group, group_info)
        return group_data_map

    def _calculate_and_group(self,
                             program: Program,
                             group: Group,
                             sites: FrozenSet[Site],
                             night_indices: NightIndices,
                             starting_time_slots: StartingTimeslots,
                             night_configurations: NightConfigurationData,
                             ranker: Ranker,
                             group_data_map: GroupDataMap) -> GroupDataMap:
        """
        Calculate the GroupInfo for an AND group that contains subgroups and add it to
        the group_data_map.
        """
        if not isinstance(group, AndGroup):
            raise ValueError(f'Tried to process group {group.id} as an AND group.')
        if isinstance(group.children, Observation):
            raise ValueError(f'Tried to process observation group {group.id} as an AND group.')

        # Process all subgroups and then process this group directly.
        # Ignore the return values here: they will just accumulate in group_info_map.
        for subgroup in group.children:
            self._calculate_group(program, subgroup, sites, night_indices, starting_time_slots, night_configurations,
                                  ranker, group_data_map)

        # We can only schedule this group if its sites are all being scheduled; however, we still want to
        # score this group's children if their sites are covered: hence the check after the child scoring.
        if not group.sites().issubset(sites):
            logger.warning(f'Cannot score group {group.id}: contains observations in sites not being scheduled.')
            return group_data_map

        # TODO: Confirm that this is correct behavior.
        # Make sure that there is an entry for each subgroup. If not, we skip.
        if any(sg.unique_id not in group_data_map for sg in group.children):
            # We don't include the root group for scheduling, so if it is missing children, we don't output a warning.
            if group.id != ROOT_GROUP_ID:
                missing_subgroups = [sg.unique_id.id for sg in group.children if sg.unique_id not in group_data_map]
                missing_str = ', '.join(missing_subgroups)
                logger.warning(f'Selector skipping group {group.unique_id}: scores missing for children: '
                               f'{missing_str}.')
            return group_data_map

        # Calculate the most restrictive conditions.
        subgroup_conditions = ([group_data_map[sg.unique_id].group_info.minimum_conditions for sg in group.children])
        mrc = Conditions.most_restrictive_conditions(subgroup_conditions)

        # This group will always be splittable unless we have some bizarre nesting.
        is_splittable = len(group.observations()) > 1 or len(group.observations()[0].sequence) > 1

        # TODO: Do we need standards?
        # TODO: This is not how we handle standards. Fix this.
        # standards = np.sum([group_info_map[sg.unique_id].standards for sg in group.children])
        standards = 0.

        # The group is filtered in for a night_idx if all its subgroups are filtered in for that night_idx.
        night_filtering = {night_idx: all(
                                  group_data_map[sg.unique_id].group_info.night_filtering[night_idx]
                                  for sg in group.children
        ) for night_idx in night_indices}

        # The conditions score is the product of the conditions scores for each subgroup across each night.
        conditions_score = {}
        for night_idx in night_indices:
            conditions_scores_for_night = [group_data_map[sg.unique_id].group_info.conditions_score[night_idx]
                                           for sg in group.children]
            conditions_score[night_idx] = np.multiply.reduce(conditions_scores_for_night)

        # The wind score is the product of the wind scores for each subgroup across each night.
        wind_score = {}
        for night_idx in night_indices:
            wind_scores_for_night = [group_data_map[sg.unique_id].group_info.wind_score[night_idx]
                                     for sg in group.children]
            wind_score[night_idx] = np.multiply.reduce(wind_scores_for_night)

        # The schedulable slot indices are the unions of the schedulable slot indices for each subgroup
        # across each night.
        schedulable_slot_indices = {
            night_idx:
            # For each night, take the concatenation of the schedulable time slots for all children of the group
            # and make it unique, which also puts it in sorted order.
            np.unique(np.concatenate([
                group_data_map[sg.unique_id].group_info.schedulable_slot_indices[night_idx]
                for sg in group.children
            ]))
            for night_idx in night_indices
        }

        # Calculate the scores for the group across all nights across all timeslots.
        scores = ranker.score_group(group, group_data_map)

        group_info = GroupInfo(
            minimum_conditions=mrc,
            is_splittable=is_splittable,
            standards=standards,
            night_filtering=night_filtering,
            conditions_score=conditions_score,
            wind_score=wind_score,
            schedulable_slot_indices=schedulable_slot_indices,
            scores=scores
        )

        group_data_map[group.unique_id] = GroupData(group, group_info)
        return group_data_map

    def _calculate_or_group(self,
                            program: Program,
                            group: Group,
                            sites: FrozenSet[Site],
                            night_indices: NightIndices,
                            starting_time_slots: StartingTimeslots,
                            night_configurations: NightConfigurationData,
                            ranker: Ranker,
                            group_data_map: GroupDataMap) -> GroupDataMap:
        """
        Calculate the GroupInfo for an AND group that contains subgroups and add it to
        the group_data_map.

        Not yet implemented.
        """
        raise NotImplementedError(f'Selector does not yet handle OR groups: {group.id}')

    @staticmethod
    def _wind_conditions(variant: Variant,
                         azimuth: Angle) -> npt.NDArray[float]:
        """
        Calculate the effect of the wind conditions on the score of an observation.
        """
        wind = np.ones(len(azimuth))
        az_wd = np.abs(azimuth.to_value() - variant.wind_dir.to_value())
        idx = np.where(np.logical_and(variant.wind_spd > Selector._wind_spd_bound,  # * u.m / u.s
                                      np.logical_or(az_wd <= Selector._wind_sep.to_value(),
                                                    360 - az_wd <= Selector._wind_sep.to_value())))[0]

        # Adjust down to 0 if the wind conditions are not adequate.
        wind[idx] = 0
        return wind

    @staticmethod
    def _match_conditions(required_conditions: Conditions,
                          actual_conditions: Variant,
                          neg_ha: bool,
                          too_status: Optional[TooType]) -> npt.NDArray[float]:
        """
        Determine if the required conditions are satisfied by the actual conditions variant.
        * required_conditions: the conditions required by an observation
        * actual_conditions: the actual conditions variant, which can hold scalars or numpy arrays
        * neg_ha: a numpy array indexed by night that indicates if the first angle hour is negative
        * too_status: the TOO status of the observation, if any

        We return a numpy array with entries in [0,1] indicating how well the actual conditions match
        the required conditions.

        Note that:
        * 0 indicates that the actual conditions do not satisfy the required ones;
        * 1 indicates that the actual conditions perfectly satisfy the required ones, or we do not care, as is the
            case in certain types of targets of opportunity; and
        * a value in (0,1) indicates that the actual conditions over-satisfy the required ones.

        TODO: Check this for correctness.
        """
        # TODO: Can we move part of this to the mini-model? Do we want to?

        # Convert the actual conditions to arrays.
        actual_iq = np.asarray(actual_conditions.iq)
        actual_cc = np.asarray(actual_conditions.cc)

        # The mini-model manages the IQ, CC, WV, and SB being the same types and sizes
        # so this check is a bit extraneous: if one has a ndim of 0, they all will.
        scalar_input = actual_iq.ndim == 0 or actual_cc.ndim == 0
        if actual_iq.ndim == 0:
            actual_iq = actual_iq[None]
        if actual_cc.ndim == 0:
            actual_cc = actual_cc[None]

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

        bad_cond_idx = np.where(np.logical_or(bad_iq, bad_cc))[0]
        cond_match = np.ones(length)
        cond_match[bad_cond_idx] = 0

        # Penalize for using IQ / CC that is better than needed:
        # Multiply the weights by actual value / value where value is better than required and target
        # does not set soon and is not a rapid ToO.
        # This should work as we are adjusting structures that are passed by reference.
        def adjuster(array, value):
            better_idx = np.where(array < value)[0] if neg_ha else np.array([])
            if len(better_idx) > 0 and (too_status is None or too_status not in {TooType.RAPID, TooType.INTERRUPT}):
                cond_match[better_idx] = cond_match[better_idx] * array[better_idx] / value

        adjuster(actual_iq, required_conditions.iq)
        adjuster(actual_cc, required_conditions.cc)

        if scalar_input:
            cond_match = np.squeeze(cond_match)
        return cond_match
