# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Dict, FrozenSet, List, Optional, Set

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import Angle
from lucupy.minimodel import (ALL_SITES, AndGroup, Conditions, Group, Observation, ObservationClass, ObservationStatus,
                              ProgramID, ROOT_GROUP_ID, Site, TooType, NightIndex, UniqueGroupID, Variant)

from scheduler.core.calculations import GroupData, GroupDataMap, GroupInfo, ProgramInfo, Selection
from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.collector import Collector
from scheduler.core.components.ranker import DefaultRanker, Ranker
from scheduler.core.builder.blueprint import Blueprints
from scheduler.services import logger_factory
from scheduler.services.resource import NightConfiguration
ENV = Blueprints.sources.environment

logger = logger_factory.create_logger(__name__)

# Aliases to pass around resource availability information for sites and night indices.
NightConfigurations = Dict[NightIndex, NightConfiguration]
NightConfigurationData = Dict[Site, NightConfigurations]


@dataclass(frozen=True)
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

    _wind_sep: ClassVar[Angle] = 20. * u.deg

    # TODO BRYAN: store the group info map of info for all groups.
    _group_info_map: Dict[UniqueGroupID, GroupInfo] = field(init=False, repr=False, compare=False)

    @staticmethod
    def _get_top_level_groups(group_data_map: GroupDataMap) -> List[GroupData]:
        """
        Given a GroupDataMap for a program, what we want is:
        The subset of groups that we are interested in is the swt of groups whose parent
        group (not including the root) does not appear in the groups here.
        """
        # Get all scheduling groups excluding the root.
        scheduling_groups = [group for (group, _) in group_data_map.values()
                             if group.id != ROOT_GROUP_ID and group.is_scheduling_group()]

        # Extract the children's names of all scheduling groups except the root into a set.
        # This way, we only have to check IDs for equality instead of the entire group structure.
        scheduling_group_children_names = {
            child.id for group in scheduling_groups for child in group.children
        }

        # Find the group_data for groups that are not the root group and are not in any scheduling_group.
        return [group_data for group_data in group_data_map.values()
                if (group_data.group.id != ROOT_GROUP_ID and
                    group_data.group.id not in scheduling_group_children_names)]

    def select(self,
               sites: FrozenSet[Site] = ALL_SITES,
               night_indices: Optional[npt.NDArray[NightIndex]] = None,
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

        TODO: Currently OR groups cannot be handled. They are not supported by OCS and
        TODO: attempts to include them will result in a NotImplementedException.
        TODO: Further design work must be done to determine how to score them and how to
        TODO: determine the time slots at which they can be scheduled.
        """
        # TODO BRYAN: wants all of the group info available.
        _group_info_map: Dict[UniqueGroupID, GroupInfo] = {}

        # If no night indices are specified, assume all night indices.
        if night_indices is None:
            night_indices = np.arange(len(self.collector.time_grid))

        # The number of nights in the plan.
        num_nights_plan = len(night_indices)

        # If no manual ranker was specified, create the default.
        if ranker is None:
            ranker = DefaultRanker(self.collector, night_indices)

        # The night_indices in the Selector and Ranker must be the same.
        if not np.array_equal(night_indices, ranker.night_indices):
            raise ValueError(f'The Ranker must have the same night indices as the Selector select method.')

        # Get the night configuration for all nights.
        night_configurations = {site: self.collector.night_configurations(site, night_indices) for site in sites}

        # Create the structure to hold the mapping fom program ID to its group info.
        program_info: Dict[ProgramID, ProgramInfo] = {}

        # A flat top-level list of GroupData indexed by UniqueGroupID.
        schedulable_groups: Dict[UniqueGroupID, GroupData] = {}

        # Observation group IDs.
        obs_group_ids: Set[UniqueGroupID] = set()

        for program_id in Collector.get_program_ids():
            program = Collector.get_program(program_id)
            if program is None:
                logger.error(f'Program {program_id} was not found in the Collector.')
                continue

            # TODO: We have to check across nights.
            # Calculate the group info and put it in the structure if there is actually group
            # info data inside it, i.e. feasible time slots for it in the plan.
            # We must check across all nights, hence the second for.
            # This will filter out all GroupInfo objects that do not have schedulable slots.
            unfiltered_group_data_map = self._calculate_group(program.root_group,
                                                              sites,
                                                              night_indices,
                                                              night_configurations,
                                                              ranker)

            # TODO BRYAN: keep unfiltered group info.
            for group_data in unfiltered_group_data_map.values():
                _group_info_map[group_data.group.unique_id()] = group_data.group_info

            group_data_map = {gp_id: gp_data for gp_id, gp_data in unfiltered_group_data_map.items()
                              if any(len(indices) > 0 for indices in gp_data.group_info.schedulable_slot_indices)}

            # Find the IDs of the observation groups.
            obs_group_ids.update(gp_id for gp_id, gp_data in group_data_map if gp_data.group.is_observation_group())

            # This filters out any programs that have no groups with any schedulable slots.
            if group_data_map:
                # Get the top-level groups (excluding root) in group_data_map and add to the schedulable_groups map.
                top_level_group_data = Selector._get_top_level_groups(group_data_map)
                for group_data in top_level_group_data:
                    schedulable_groups[group_data.group.unique_id()] = group_data

                # Remember that in an observation group, the only child is an Observation: hence references here
                # to group.children are simply the Observation.
                observations = {group_data.group.children.id: group_data.group.children
                                for group_data in group_data_map.values()
                                if group_data.group.is_observation_group()}
                target_info = {obs_id: self.collector.get_target_info(obs_id) for obs_id, obs in observations.items()}
                program_info[program.id] = ProgramInfo(
                    program=deepcopy(program),
                    group_data=group_data_map,
                    observations=deepcopy(observations),
                    target_info=target_info
                )

        # TODO BRYAN: store the group info map of info for all groups.
        object.__setattr__(self, '_group_info_map', _group_info_map)

        # The end product is a map of ProgramID to a map of GroupID to GroupInfo, where
        # at least one GroupInfo has schedulable slots.
        return Selection(
            program_info=program_info,
            night_events={site: self.collector.get_night_events(site) for site in sites},
            schedulable_groups=schedulable_groups,
            obs_group_ids=frozenset(obs_group_ids),
            num_nights_plan=num_nights_plan,
            time_slot_length=self.collector.time_slot_length.to_datetime()
        )

    def get_group_info(self, unique_group_id: UniqueGroupID) -> Optional[GroupInfo]:
        """
        Check to see if the group_id has group_info associated with it, and if so, return it.
        Else return None.
        """
        logger.warning('Selector.get_group_info should NOT be used in production code.')
        return self._group_info_map[unique_group_id]

    def _calculate_group(self,
                         group: Group,
                         sites: FrozenSet[Site],
                         night_indices: npt.NDArray[NightIndex],
                         night_configurations: NightConfigurationData,
                         ranker: Ranker,
                         group_data_map: GroupDataMap = None) -> GroupDataMap:
        """
        Delegate this group to the proper calculation method.
        """
        if group_data_map is None:
            group_data_map = {}

        if group.is_observation_group():
            processor = self._calculate_observation_group
        elif group.is_and_group():
            processor = self._calculate_and_group
        elif group.is_or_group():
            processor = self._calculate_or_group
        else:
            raise ValueError(f'Could not process group {group.id}')

        return processor(group, sites, night_indices, night_configurations, ranker, group_data_map)

    def _calculate_observation_group(self,
                                     group: Group,
                                     sites: FrozenSet[Site],
                                     night_indices: npt.NDArray[NightIndex],
                                     night_configurations: NightConfigurationData,
                                     ranker: Ranker,
                                     group_data_map: GroupDataMap) -> GroupDataMap:
        """
        Calculate the GroupInfo for a group that contains an observation and add it to
        the group_data_map.

        TODO: Not every observation has TargetInfo: if it is not in a specified class, it will not.
        TODO: How do we handle these cases? For now, I am going to skip any observation with a missing TargetInfo.
        """
        if not group.is_observation_group():
            raise ValueError(f'Non-observation group {group.id} cannot be treated as observation group.')

        obs = group.children

        # TODO: Do we really want to include OBSERVED here?
        if obs.status not in {ObservationStatus.ONGOING, ObservationStatus.READY, ObservationStatus.OBSERVED}:
            return group_data_map
        if obs.site not in sites:
            logger.warning(f'Selector skipping observation {obs.id} as not in a designated site.')
            return group_data_map

        # We ignore the Observation if:
        # 1. There is no target info associated with it.
        target_info = Collector.get_target_info(obs.id)
        if target_info is None:
            return group_data_map
        # 2. There are no constraints associated with it.
        if obs.constraints is None:
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
        night_filtering = np.array([night_configurations[obs.site][night_idx].filter.group_filter(group)
                                    for night_idx in night_indices])

        if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
            # If we are science or progcal, then the neg HA value for a night is if the first HA for the night
            # is negative.
            neg_ha = np.array([target_info[night_idx].hourangle[0].value < 0 for night_idx in ranker.night_indices])
        else:
            neg_ha = np.array([False] * len(ranker.night_indices))
        too_type = obs.too_type

        # Calculate when the conditions are met and an adjustment array if the conditions are better than needed.
        # TODO: Maybe we only need to concern ourselves with the night indices where the resources are in place.
        conditions_score = []
        wind_score = []

        # We need the night_events for the night for timing information.
        night_events = self.collector.get_night_events(obs.site)

        for night_idx in ranker.night_indices:
            # TODO: Use Selector._env when the get actual conditions variant method doesn't return static data.
            # actual_conditions = Selector._env.get_actual_conditions_variant(obs.site, night_events.times[night_idx])
            actual_conditions = ENV.get_actual_conditions_variant(obs.site, night_events.times[night_idx])

            # If we can obtain the conditions variant, calculate the conditions and wind mapping.
            # Otherwise, use arrays of all zeros to indicate that we cannot calculate this information.
            if actual_conditions is not None:
                conditions_score.append(Selector._match_conditions(mrc, actual_conditions, neg_ha[night_idx], too_type))
                wind_score.append(Selector._wind_conditions(actual_conditions, target_info[night_idx].az))
            else:
                zero = np.zeros(len(night_events.times[night_idx]))
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
            if night_filtering[night_idx]:
                schedulable_slot_indices.append(np.where(conditions_score[night_idx][vis_idx] > 0)[0])
            else:
                schedulable_slot_indices.append(np.array([]))

        # Calculate the scores for the observation across all nights across all timeslots.
        # Multiply by the conditions score to adjust the scores.
        # Note that np.multiply will handle lists of numpy arrays.
        # This generates a warning about ragged array deprecation, but seems to produce the right shape of structure.
        scores = np.multiply(np.multiply(conditions_score, ranker.get_observation_scores(obs.id)), wind_score)

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

        group_data_map[group.id] = GroupData(deepcopy(group), group_info)
        return group_data_map

    def _calculate_and_group(self,
                             group: Group,
                             sites: FrozenSet[Site],
                             night_indices: npt.NDArray[NightIndex],
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
            self._calculate_group(subgroup, sites, night_indices, night_configurations, ranker, group_data_map)

        # TODO: Confirm that this is correct behavior.
        # Make sure that there is an entry for each subgroup. If not, we skip.
        if any(sg.id not in group_data_map for sg in group.children):
            return group_data_map

        # Calculate the most restrictive conditions.
        subgroup_conditions = ([group_data_map[sg.id].group_info.minimum_conditions for sg in group.children])
        mrc = Conditions.most_restrictive_conditions(subgroup_conditions)

        # This group will always be splittable unless we have some bizarre nesting.
        is_splittable = len(group.observations()) > 1 or len(group.observations()[0].sequence) > 1

        # TODO: Do we need standards?
        # TODO: This is not how we handle standards. Fix this.
        # standards = np.sum([group_info_map[sg.id].standards for sg in group.children])
        standards = 0.

        # The filtering for this group is the product of filtering for the subgroups.
        sg_night_filtering = [group_data_map[sg.id].group_info.night_filtering for sg in group.children]
        night_filtering = np.multiply.reduce(sg_night_filtering).astype(bool)

        # The conditions score is the product of the conditions scores for each subgroup across each night.
        conditions_score = []
        for night_idx in night_indices:
            conditions_scores_for_night = [group_data_map[sg.id].group_info.conditions_score[night_idx]
                                           for sg in group.children]
            conditions_score.append(np.multiply.reduce(conditions_scores_for_night))

        # The wind score is the product of the wind scores for each subgroup across each night.
        wind_score = []
        for night_idx in night_indices:
            wind_scores_for_night = [group_data_map[sg.id].group_info.wind_score[night_idx]
                                     for sg in group.children]
            wind_score.append(np.multiply.reduce(wind_scores_for_night))

        # The schedulable slot indices are the unions of the schedulable slot indices for each subgroup
        # across each night.
        schedulable_slot_indices = [
            # For each night, take the concatenation of the schedulable time slots for all children of the group
            # and make it unique, which also puts it in sorted order.
            np.unique(np.concatenate([
                group_data_map[sg.id].group_info.schedulable_slot_indices[night_idx]
                for sg in group.children
            ]))
            for night_idx in night_indices
        ]

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

        group_data_map[group.id] = GroupData(deepcopy(group), group_info)
        return group_data_map

    def _calculate_or_group(self,
                            group: Group,
                            site: FrozenSet[Site],
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
        az_wd = np.abs(azimuth - variant.wind_dir)
        idx = np.where(np.logical_and(variant.wind_spd > 10,  # * u.m / u.s
                                      np.logical_or(az_wd <= Selector._wind_sep,
                                                    360. * u.deg - az_wd <= Selector._wind_sep)))[0]

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
            # TODO: We get here and array has size 1, neg_ha has size 3 (based on night_indices), and we get [0,1,2].
            # TODO: This will not work.
            # TODO EXAMPLE:
            # np.where(np.logical_and(np.array([0]) < 1, np.array([True, True, True])))
            # Out: (array([0, 1, 2]),)
            # better_idx = np.where(np.logical_and(array < value, neg_ha))[0]
            # better_tmp_idx = np.where(array < value)[0]
            # better_idx = np.where(neg_ha[better_tmp_idx])[0]
            better_idx = np.where(array < value)[0] if neg_ha else np.array([])
            if len(better_idx) > 0 and (too_status is None or too_status not in {TooType.RAPID, TooType.INTERRUPT}):
                cond_match[better_idx] = cond_match[better_idx] * array / value

        adjuster(actual_iq, required_conditions.iq)
        adjuster(actual_cc, required_conditions.cc)

        if scalar_input:
            cond_match = np.squeeze(cond_match)
        return cond_match
