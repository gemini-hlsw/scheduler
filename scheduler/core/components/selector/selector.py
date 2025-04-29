# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from dataclasses import dataclass, field
from typing import final, ClassVar, Dict, FrozenSet, Optional, TypeAlias

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import Angle
from astropy.units import Quantity
from lucupy.helpers import is_contiguous
from lucupy.minimodel import (Group, Conditions, Group, Observation, ObservationClass, ObservationStatus, Program,
                              ProgramID, ROOT_GROUP_ID, Site, TooType, NightIndex, NightIndices, TimeslotIndex,
                              UniqueGroupID, Variant, VariantSnapshot)
from lucupy.minimodel import CloudCover, ImageQuality
from lucupy.timeutils import time2slots

from scheduler.core.calculations import GroupData, GroupDataMap, GroupInfo, ProgramCalculations, ProgramInfo, Selection
from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.collector import Collector
from scheduler.core.components.ranker import DefaultRanker, Ranker
from scheduler.core.components.selector.timebuffer import TimeBuffer
from scheduler.core.types import StartingTimeslots
from scheduler.services import logger_factory
from scheduler.services.resource import NightConfiguration


__all__ = [
    'NightConfiguration',
    'NightConfigurationData',
    'Selector',
]


# Aliases to pass around resource availability information for sites and night indices.
NightConfigurations: TypeAlias = Dict[NightIndex, NightConfiguration]
NightConfigurationData: TypeAlias = Dict[Site, NightConfigurations]

logger = logger_factory.create_logger(__name__)


@final
@dataclass
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
    time_buffer: TimeBuffer

    # Store the current VariantSnapshot at each site.
    # TODO: We will use wind dir and speed later, and perhaps also WV.
    # These start off empty, but will reset at the beginning of each night by the event loop.
    _variant_snapshot_per_site: Dict[Site, VariantSnapshot] = field(init=False, default_factory=dict)

    _wind_sep: ClassVar[Angle] = 20. * u.deg
    _wind_spd_bound: ClassVar[Quantity] = 10. * u.m / u.s

    # Default values for resetting variants.
    _default_iq: ClassVar[ImageQuality] = ImageQuality.IQ70
    _default_cc: ClassVar[CloudCover] = CloudCover.CC70
    _default_wind_dir: ClassVar[Angle] = Angle(0., unit=u.deg)
    _default_wind_spd: ClassVar[Quantity] = 0.0 * (u.m / u.s)

    _default_variant_snapshot: ClassVar[VariantSnapshot] = VariantSnapshot(iq=_default_iq,
                                                                           cc=_default_cc,
                                                                           wind_dir=_default_wind_dir,
                                                                           wind_spd=_default_wind_spd)

    def __post_init__(self):
        # Make sure the number of nights to schedule is not larger than the number of nights used in visibility
        # calculations.
        if (self.num_nights_to_schedule < 0 or
                self.num_nights_to_schedule > self.collector.num_nights_calculated):
            raise ValueError(f'Scheduling requested for {self.num_nights_to_schedule} nights, but visibility '
                             f'calculations only performed for {self.collector.num_nights_calculated}. '
                             'Cannot proceed.')

        # Calculate the blocked indices per night that the Scheduler knows in advance, e.g. engineering tasks.
        self._blocked_timeslots = self._calculate_blocked_timeslots()

        self.night_configurations = {}
        for site in self.collector.sites:
            self.night_configurations[site] = self.collector.night_configurations(
                site, np.arange(self.collector.num_nights_calculated))

    @staticmethod
    def _process_starting_time_slots(sites: FrozenSet[Site],
                                     night_indices: NightIndices,
                                     starting_time_slots: Optional[StartingTimeslots]) -> StartingTimeslots:
        """
        Determine the starting timeslots for each night. These should typically be 0 unless we are only calculating
        a fraction of a night, e.g. if a new selection is being done for part of the night.
        """
        if starting_time_slots is None:
            starting_time_slots = {}

        # Make sure we have an entry for all relevant sites for all night indices.
        # Add starting index of 0 if there is no data.
        for site in sites:
            night_dict = starting_time_slots.setdefault(site, {})
            night_dict.update({night_idx: 0 for night_idx in night_indices if night_idx not in night_dict})

            # Check for extra keys.
            if extra_keys := night_dict.keys() - night_indices:
                logger.warning(f'Extra night indices for site {site.name} for starting_time_slots: {extra_keys}')

        return starting_time_slots

    def _calculate_blocked_timeslots(self) -> Dict[Site, Dict[NightIndex, npt.NDArray[TimeslotIndex]]]:
        """
        For each site, calculate the blocked timeslots for each night.
        This information comes from the Engineering Tasks, but it may be expanded in the future to contain more
        information.

        INFO: This demonstrates the calculations from time to timeslot in a robust way, given that one has access to
        the NightEvents, the Events, and the night_index.
        The NightEvents.local_dt_to_time_coords sometimes borks, as is the case here.
        """
        sites = self.collector.sites

        # TODO: This seems like an over-calculation: we only need to calculate for the number of nights we are
        # TODO: scheduling and not the visibility period. Possible room for improvement.
        night_indices = np.arange(len(self.collector.time_grid))
        time_slot_length = self.collector.time_slot_length.to_datetime()

        blocked_indices_by_site = {site: {} for site in sites}
        for site in sites:
            night_events = self.collector.get_night_events(site)
            night_configurations = self.collector.night_configurations(site, night_indices)

            blocked_indices_by_night = {}
            for night_idx in night_indices:
                # Twilights for night_idx.
                earliest_time = night_events.local_times[night_idx][0]
                latest_time = night_events.local_times[night_idx][-1]

                # Ideally, dtype would be TimeslotIndex, but numpy borks on this.
                blocked_timeslot_indices = np.array([], dtype=int)

                eng_tasks = night_configurations[night_idx].eng_tasks
                for eng_task in eng_tasks:
                    # Bound the start_time and end_time by the twilights.
                    start_time = max(eng_task.start_time, earliest_time)
                    start_delta = start_time - earliest_time
                    start_timeslot_idx = time2slots(time_slot_length, start_delta)

                    # This doesn't work, but not removing yet as we should know this is unreliable.
                    # start_indices = night_events.local_dt_to_time_coords(start_time)
                    # if start_indices is None:
                    #     logger.error(f'Engineering task {eng_task} does not have a valid start time: '
                    #                  f'determined: {start_time}, latest possible: {earliest_time}')
                    #     continue
                    # start_night_idx, start_timeslot_idx = start_indices

                    end_time = min(eng_task.end_time, latest_time)
                    end_delta = end_time - earliest_time
                    end_timeslot_idx = time2slots(time_slot_length, end_delta)

                    # This doesn't work, but not removing yet as we should know this is unreliable.
                    # end_indices = night_events.local_dt_to_time_coords(end_time)
                    # if end_indices is None:
                    #     logger.error(f'Engineering task {eng_task} does not have a valid end time: '
                    #                  f'determined: {end_time}, latest possible: {latest_time}')
                    #     continue
                    # end_night_idx, end_timeslot_idx = end_indices

                    # if start_night_idx != night_idx or end_night_idx != night_idx:
                    #     raise ValueError(f'Calculating blocked slots for {eng_task} spans multiple nights: '
                    #                      f'{start_night_idx} to {end_night_idx}, should be {night_idx}.')
                    # blocked_timeslot_indices = np.union1d(blocked_timeslot_indices,
                    #                                       np.arange(start_timeslot_idx, end_timeslot_idx))

                    # Continue to take the union of the time slots that are blocked off for the site for the night_idx.
                    blocked_timeslot_indices = np.union1d(blocked_timeslot_indices,
                                                          np.arange(start_timeslot_idx, end_timeslot_idx))
                blocked_indices_by_night[night_idx] = blocked_timeslot_indices
            blocked_indices_by_site[site] = blocked_indices_by_night
        return blocked_indices_by_site

    def update_site_variant(self,
                            site: Site,
                            variant_snapshot: Optional[VariantSnapshot] = None) -> None:
        """
        Extract the CC and IQ values from the new conditions and update them for the given site.
        """
        if site not in self.collector.sites:
            raise ValueError(f'Selector trying to update conditions for invalid site: {site.name}')
        if variant_snapshot is None:
            variant_snapshot = Selector._default_variant_snapshot
        self._variant_snapshot_per_site[site] = variant_snapshot

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
            night_conditions=self._variant_snapshot_per_site,
            starting_time_slots=starting_time_slots,
            time_slot_length=self.collector.time_slot_length.to_datetime(),
            ranker=ranker,
            _program_scorer=self.score_program
        )

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
        # Check if there is any time left for the program, allowing for the time buffer. If not, skip it.
        if program.program_awarded() + self.time_buffer(program) <= program.program_used():
            logger.debug(f'Program {program.id.id} out of time: skipping.')
            return None

        verbose = False

        if verbose:
            print(f'score_program: {program.id.id}')
            print(f'sites: {sites}, night_indices: {night_indices}, starting_time_slots: {starting_time_slots}')

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
                          if any(indices.size > 0 for indices in gp_data.group_info.schedulable_slot_indices.values())
                          and gp_data.group.id != ROOT_GROUP_ID}
        if verbose:
            print(f'group_data_map keys: {group_data_map.keys()}')
            for gp_id, gp_data in unfiltered_group_data_map.items():
                print(gp_id, gp_data.group_info.schedulable_slot_indices.values())

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

        verbose = False

        obs = group.children
        if verbose:
            print(f'\t\t\t_calculate_observation_group: {obs.unique_id.id} {obs.status}')
        if obs.status in {ObservationStatus.OBSERVED, ObservationStatus.INACTIVE, ObservationStatus.ON_HOLD}:
            logger.debug(f'Observation {obs.id.id} has a status of {obs.status.name}. Skipping.')
            return group_data_map

        if obs.status not in {ObservationStatus.READY, ObservationStatus.ONGOING}:
            raise ValueError(f'Observation {obs.id.id} has a status of {obs.status.name}.')

        # This should never happen.
        if obs.site not in sites:
            logger.debug(f'Selector ignoring request to score {obs.id}: not at a currently selected site.')
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

        # Calculate a numpy array of bool indexed by night to determine when the group can be added to the plan
        # based on the night configuration filtering.
        # TODO: We also filter on program here, but this would be better done in score_program.
        # TODO: That would require some thought as to how to do this there given the structure of a Selection.
        night_filtering: Dict[NightIndex, bool] = {}
        for night_idx in night_indices:
            night_filter = night_configurations[obs.site][night_idx].filter
            # NOTE: to only do group filtering, comment out the first line and use the second line.
            night_filtering[night_idx] = night_filter.program_filter(program) and night_filter.group_filter(group)
            # night_filtering[night_idx] = night_filter.group_filter(group)

        if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
            # If we are science or progcal, then the check if the first HA for the night is negative,
            # indicating that the target is rising
            rising = {night_idx: target_info[night_idx].hourangle[0].value < 0 for night_idx in night_indices}
        else:
            rising = {night_idx: True for night_idx in night_indices}
        too_type = obs.too_type

        # Calculate when the conditions are met and an adjustment array if the conditions are better than needed.
        # TODO: Maybe we only need to concern ourselves with the night indices where the resources are in place.
        conditions_score = {}
        wind_score = {}

        # We need the night_events for the night for timing information.
        night_events = self.collector.get_night_events(obs.site)

        for night_idx in night_indices:
            # Get the conditions for the night. We need values for every timeslot, but at the end, we will
            # zero out the timeslots that have already passed.
            total_timeslots_in_night = len(night_events.times[night_idx])
            starting_timeslot_in_night = starting_time_slots[obs.site][night_idx]

            # Determine how closely the matched the required conditions are to the actual conditions.
            # This creates a Variant spanning the whole night when part of the night up to starting_timeslot_in_night
            # has already been covered and should be ineligible for scheduling.
            # Zero out the part of the night that was already done.
            variant = self._variant_snapshot_per_site[obs.site].make_variant(total_timeslots_in_night)
            # print(f'Selector: Night {night_idx} for obs {obs.id.id} ({obs.internal_id}) @ {obs.site.name}')
            # print(f'Current conditions: {max(variant.iq)} {max(variant.cc)} {max(variant.wind_dir)} {max(variant.wind_spd)}')
            # print(f'Conditions req: IQ {mrc.iq}, CC {mrc.cc}')
            # print(f'rising: {rising[night_idx]}, Too: {too_type}')
            conditions_score[night_idx] = Selector.match_conditions(mrc, variant, rising[night_idx], too_type)
            conditions_score[night_idx][:starting_timeslot_in_night] = 0
            wind_score[night_idx] = Selector._wind_conditions(variant, target_info[night_idx].az)
            # print(f'conditions score: {max(conditions_score[night_idx])}, wind_score: {max(wind_score[night_idx])}')

        # Calculate the schedulable slot indices.
        # These are the indices where the observation has:
        # 1. Visibility
        # 2. Resources available
        # 3. Conditions that are met
        schedulable_slot_indices = {}
        for night_idx in night_indices:
            vis_idx = target_info[night_idx].visibility_slot_idx
            # print(f'len(vis_idx) = {len(vis_idx)}')
            if night_filtering[night_idx]:
                schedulable_slot_indices[night_idx] = np.where(conditions_score[night_idx][vis_idx] > 0)[0]
            else:
                schedulable_slot_indices[night_idx] = np.array([])
        # print(f'number schedulable slots night: {len(schedulable_slot_indices[night_idx])}')

        obs_scores, obs_metrics = ranker.score_observation(program, obs, self.night_configurations)
        # print(f'obs_scores: {max(obs_scores[night_idx])}')

        # Calculate the scores for the observation across all night indices across all timeslots.
        scores = {night_idx: np.multiply(
                np.multiply(conditions_score[night_idx], obs_scores[night_idx]),
                wind_score[night_idx]) for night_idx in night_indices}

        # Zero out the scores for the blocked timeslots.
        blocked_timeslots = self._blocked_timeslots[obs.site]
        for night_idx in night_indices:
            scores[night_idx][blocked_timeslots[night_idx]] = 0.0

        # Zero out the data for each night index's starting time slots prior to the value specified (if specified)
        # and the night index was included in scoring.
        starting_time_slots_for_site = starting_time_slots[obs.site]
        for night_idx, time_slot_idx in starting_time_slots_for_site.items():
            if night_idx in night_indices:
                scores[night_idx][:time_slot_idx] = 0.0
        # print(f'scores: {max(scores[night_idx])}\n')

        # These scores might differ from the observation score in the ranker since they have been adjusted for
        # conditions and wind.
        group_info = GroupInfo(
            minimum_conditions=mrc,
            is_splittable=is_splittable,
            night_filtering=night_filtering,
            conditions_score=conditions_score,
            wind_score=wind_score,
            schedulable_slot_indices=schedulable_slot_indices,
            metrics=obs_metrics,
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
        if not isinstance(group, Group):
            raise ValueError(f'Tried to process group {group.id} as an AND group.')
        if isinstance(group.children, Observation):
            raise ValueError(f'Tried to process observation group {group.id} as an AND group.')

        verbose = False

        if verbose:
            print(f'\t_calculate_and_group: {group.unique_id.id}')

        # Process all subgroups and then process this group directly.
        # Ignore the return values here: they will just accumulate in group_info_map.
        for subgroup in group.children:
            if verbose:
                print(f'\t\tsubgroup: {subgroup.unique_id.id}')
            self._calculate_group(program, subgroup, sites, night_indices, starting_time_slots, night_configurations,
                                  ranker, group_data_map)

        # We can only schedule this group if its sites are all being scheduled; however, we still want to
        # score this group's children if their sites are covered: hence the check after the child scoring.
        if not group.sites().issubset(sites):
            logger.warning(f'Cannot score group {group.id}: contains observations in sites not being scheduled.')
            return group_data_map

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
        # Type checker is balking here at the list of npt.NDArray[TimeslotIndex].
        # noinspection PyTypeChecker
        schedulable_slot_indices = {
            night_idx:
            # For each night, take the concatenation of the schedulable time slots for all children of the group
            # and make it unique, which also puts it in sorted order.
            np.unique(np.concatenate([
                group_data_map[sg.unique_id].group_info.schedulable_slot_indices[night_idx]
                for sg in group.children
            ]))
            # If we want intersection of schedulable nights instead of union, use this code.
            # Requires: from functools import reduce
            # np.array([reduce(np.intersect1d,
            #                  [group_data_map[sg.unique_id].group_info.schedulable_slot_indices[night_idx]
            #                   for sg in group.children])])
            for night_idx in night_indices
        }

        # Calculate the scores for the group across all nights across all timeslots.
        scores, obs_metrics = ranker.score_group(group, group_data_map)

        group_info = GroupInfo(
            minimum_conditions=mrc,
            is_splittable=is_splittable,
            night_filtering=night_filtering,
            conditions_score=conditions_score,
            wind_score=wind_score,
            schedulable_slot_indices=schedulable_slot_indices,
            metrics=obs_metrics,
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
    def match_conditions(required_conditions: Conditions,
                         actual_conditions: Variant,
                         rising: bool,
                         too_status: Optional[TooType]) -> npt.NDArray[float]:
        """
        Determine if the required conditions are satisfied by the actual conditions variant.
        * required_conditions: the conditions required by an observation
        * actual_conditions: the actual conditions variant, which can hold scalars or numpy arrays
        * rising: a numpy array indexed by night that indicates if the first angle hour is negative => rising
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
            better_idx = np.where(array < value)[0] if rising else np.array([])
            if len(better_idx) > 0 and (too_status is None or too_status not in {TooType.RAPID, TooType.INTERRUPT}):
                # cond_match[better_idx] = cond_match[better_idx] * array[better_idx] / value
                cond_match[better_idx] = cond_match[better_idx] * (1.0 - (value - array[better_idx]))

        adjuster(actual_iq, required_conditions.iq)
        adjuster(actual_cc, required_conditions.cc)

        if scalar_input:
            cond_match = np.squeeze(cond_match)
        return cond_match
