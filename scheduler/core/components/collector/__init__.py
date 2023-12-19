# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import time
from dataclasses import dataclass
from datetime import timedelta
from inspect import isclass
from typing import ClassVar, Dict, FrozenSet, Iterable, List, Optional, Tuple, Type, final

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from lucupy import sky
from lucupy.minimodel import (ALL_SITES, Constraints, ElevationType, NightIndex, NightIndices, NonsiderealTarget,
                              Observation, ObservationID, ObservationClass, Program, ProgramID, ProgramTypes, Semester,
                              SiderealTarget, Site, SkyBackground, Target, TimeslotIndex, QAState, ObservationStatus,
                              Group)
from lucupy.timeutils import time2slots

from scheduler.core.calculations import NightEvents, TargetInfo, TargetInfoMap, TargetInfoNightIndexMap
from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.nighteventsmanager import NightEventsManager
from scheduler.core.plans import Plan, Plans, Visit
from scheduler.core.programprovider.abstract import ProgramProvider
from scheduler.core.sources import Sources
from scheduler.services import logger_factory
from scheduler.services.resource import NightConfiguration
from scheduler.services.resource import ResourceService

# Set to INFO for now to record time accounting activity.
logger = logger_factory.create_logger(__name__)


# TODO: Merge this if possible with Visit.
@final
@dataclass(frozen=True)
class GroupVisits:
    """Container for holding group information for each visit"""
    group: Group
    visits: List[Visit]

    def start_time_slot(self):
        if not self.visits:
            raise RuntimeError(f'start_time_slot requested, but no visits recorded for {self.group.unique_id}')
        # return min([v.start_time_slot for v in self.visits])
        return self.visits[0].start_time_slot

    def end_time_slot(self):
        if not self.visits:
            raise RuntimeError(f'end_time_slot requested, but no visits recorder for {self.group.unique_id}')
        return self.visits[-1].start_time_slot + self.visits[-1].time_slots - 1


@final
@dataclass
class Collector(SchedulerComponent):
    """
    The interval [start_vis_time, end_vis_time] indicates the time interval that we want to consider during
    the scheduling for visibility time. Note that the generation of plans will begin on the night indicated by
    start_vis_time and proceed for num_nights_to_schedule, a parameter passed to the Selector, which must
    represent fewer nights than in the [start_vis_time, end_vis_time] schedule.

    Also note that we never have need to calculate visibility retroactively, hence why plan generation begins
    on the night of start_vis_time.

    Here, we just perform the necessary calculations, and are not concerned with the number of nights to be
    scheduled.
    """
    start_vis_time: Time
    end_vis_time: Time
    sites: FrozenSet[Site]
    semesters: FrozenSet[Semester]
    sources: Sources
    time_slot_length: TimeDelta
    program_types: FrozenSet[ProgramTypes]
    obs_classes: FrozenSet[ObservationClass]

    # Manage the NightEvents with a NightEventsManager to avoid unnecessary recalculations.
    _night_events_manager: ClassVar[NightEventsManager] = NightEventsManager()

    # Resource service.
    # TODO: This will be moved out when event processing is handled.
    _resource_service: ClassVar[ResourceService]

    # This should not be populated, but we put it here instead of in __post_init__ to eliminate warnings.
    # This is a list of the programs as read in.
    # We only want to read these in once unless the program_types change, which they should not.
    _programs: ClassVar[Dict[ProgramID, Program]] = {}

    # A set of ObservationIDs per ProgramID.
    _observations_per_program: ClassVar[Dict[ProgramID, FrozenSet[ObservationID]]] = {}

    # This is a map of observation information that is computed as the programs
    # are read in. It contains both the Observation and the base Target (if any) for
    # the observation.
    _observations: ClassVar[Dict[ObservationID, Tuple[Observation, Optional[Target]]]] = {}

    # The target information is dependent on the:
    # 1. TargetName
    # 2. ObservationID (for the associated constraints and site)
    # 4. NightIndex of interest
    # We want the ObservationID in here so that any target sharing in GPP is deliberately split here, since
    # the target info is observation-specific due to the constraints and site.
    _target_info: ClassVar[TargetInfoMap] = {}

    # Definition of a day to be used to get local dates for resources.
    _DAY: ClassVar[timedelta] = timedelta(days=1)

    # The default timeslot length currently used.
    DEFAULT_TIMESLOT_LENGTH: ClassVar[Time] = 1.0 * u.min

    # These are exclusive to the create_time_array.
    _MIN_NIGHT_EVENT_TIME: ClassVar[Time] = Time('1980-01-01 00:00:00', format='iso', scale='utc')

    # NOTE: This logs an ErfaWarning about dubious year. This is due to using a future date and not knowing
    # how many leap seconds have happened: https://github.com/astropy/astropy/issues/5809
    _MAX_NIGHT_EVENT_TIME: ClassVar[Time] = Time('2100-01-01 00:00:00', format='iso', scale='utc')

    # The number of milliarcsecs in a degree, for proper motion calculation.
    _MILLIARCSECS_PER_DEGREE: ClassVar[int] = 60 * 60 * 1000

    # Used in calculating proper motion.
    _EPOCH2TIME: ClassVar[Dict[float, Time]] = {}

    def __post_init__(self):
        """
        Initializes the internal data structures for the Collector and populates them.
        """
        # Check that the times are valid.
        if not np.isscalar(self.start_vis_time.value):
            msg = f'Illegal start time (must be scalar): {self.start_vis_time}.'
            raise ValueError(msg)
        if not np.isscalar(self.end_vis_time.value):
            msg = f'Illegal end time (must be scalar): {self.end_vis_time}.'
            raise ValueError(msg)
        if self.start_vis_time > self.end_vis_time:
            msg = f'Start time ({self.start_vis_time}) cannot occur later than end time ({self.end_vis_time}).'
            raise ValueError(msg)

        # Set up the time grid for the period under consideration in calculations: this is an astropy Time
        # object from start_time to end_time inclusive, with one entry per day.
        # Note that the format is in jdate.
        self.time_grid = Time(np.arange(self.start_vis_time.jd,
                                        self.end_vis_time.jd + 1.0, (1.0 * u.day).value),
                              format='jd')

        # The number of nights for which we are performing calculations.
        self.num_nights_calculated = len(self.time_grid)

        # TODO: This code can be greatly simplified. The night_events only have to be calculated once.
        # Create the night events, which contain the data for all given nights by site.
        # This may retrigger a calculation of the night events for one or more sites.
        self.night_events = {
            site: Collector._night_events_manager.get_night_events(self.time_grid, self.time_slot_length, site)
            for site in self.sites
        }
        Collector._resource_service = self.sources.origin.resource

    def get_night_events(self, site: Site) -> NightEvents:
        return Collector._night_events_manager.get_night_events(self.time_grid,
                                                                self.time_slot_length,
                                                                site)

    @staticmethod
    def get_program_ids() -> Iterable[ProgramID]:
        """
        Return a list of all the program IDs stored in the Collector.
        """
        return Collector._programs.keys()

    @staticmethod
    def get_program(program_id: ProgramID) -> Optional[Program]:
        """
        If a program with the given ID exists, return it.
        Otherwise, return None.
        """
        return Collector._programs.get(program_id, None)

    @staticmethod
    def get_all_observations() -> Iterable[Observation]:
        return [obs_data[0] for obs_data in Collector._observations.values()]

    @staticmethod
    def get_observation_ids(program_id: Optional[ProgramID] = None) -> Optional[Iterable[ObservationID]]:
        """
        Return the observation IDs in the Collector.
        If the prog_id is specified, limit these to those in the specified in the program.
        If no such prog_id exists, return None.
        If no prog_id is specified, return a complete list of observation IDs.
        """
        if program_id is None:
            return Collector._observations.keys()
        return Collector._observations_per_program.get(program_id, None)

    @staticmethod
    def get_observation(obs_id: ObservationID) -> Optional[Observation]:
        """
        Given an ObservationID, if it exists, return the Observation.
        If not, return None.
        """
        value = Collector._observations.get(obs_id, None)
        return None if value is None else value[0]

    @staticmethod
    def get_base_target(obs_id: ObservationID) -> Optional[Target]:
        """
        Given an ObservationID, if it exists and has a base target, return the Target.
        If one of the conditions is not met, return None.
        """
        value = Collector._observations.get(obs_id, None)
        return None if value is None else value[1]

    @staticmethod
    def get_observation_and_base_target(obs_id: ObservationID) -> Optional[Tuple[Observation, Optional[Target]]]:
        """
        Given an ObservationID, if it exists, return the Observation and its Target.
        If not, return None.
        """
        return Collector._observations.get(obs_id, None)

    @staticmethod
    def get_target_info(obs_id: ObservationID) -> Optional[TargetInfoNightIndexMap]:
        """
        Given an ObservationID, if the observation exists and there is a target for the
        observation, return the target information as a map from NightIndex to TargetInfo.
        """
        info = Collector.get_observation_and_base_target(obs_id)
        if info is None:
            return None

        obs, target = info
        if target is None:
            return None

        target_name = target.name
        return Collector._target_info.get((target_name, obs_id), None)

    @staticmethod
    def _process_timing_windows(prog: Program, obs: Observation) -> List[Time]:
        """
        Given an Observation, convert the TimingWindow information in it to a simpler format
        to verify by converting each TimingWindow representation to a collection of Time frames
        based on the start, duration, repeat, and period.

        If no timing windows are given, then create one large timing window for the entire program.

        TODO: Look into simplifying to datetime instead of AstroPy Time.
        TODO: We may want to store this information in an Observation for future use.
        """
        if not obs.constraints or len(obs.constraints.timing_windows) == 0:
            # Create a timing window for the entirety of the program.
            windows = [Time([prog.start, prog.end])]
        else:
            windows = []
            for tw in obs.constraints.timing_windows:
                t0 = time.mktime(tw.start.utctimetuple()) * 1000 * u.ms
                begin = Time(t0.to_value('s'), format='unix', scale='utc')
                duration = tw.duration.total_seconds() / 3600.0 * u.h
                repeat = max(1, tw.repeat)
                period = tw.period.total_seconds() / 3600.0 * u.h if tw.period is not None else 0.0 * u.h
                windows.extend([Time([begin + i * period, begin + i * period + duration]) for i in range(repeat)])

        return windows

    @staticmethod
    def _calculate_proper_motion(target: SiderealTarget, target_time: Time) -> SkyCoord:
        """
        Calculate the proper motion of a target.
        """
        pm_ra = target.pm_ra / Collector._MILLIARCSECS_PER_DEGREE
        pm_dec = target.pm_dec / Collector._MILLIARCSECS_PER_DEGREE
        epoch_time = Collector._EPOCH2TIME.setdefault(target.epoch, Time(target.epoch, format='jyear'))
        time_offsets = target_time - epoch_time
        new_ra = (target.ra + pm_ra * time_offsets.to(u.yr).value) * u.deg
        new_dec = (target.dec + pm_dec * time_offsets.to(u.yr).value) * u.deg
        return SkyCoord(new_ra, new_dec, frame='icrs', unit='deg')

    def _calculate_target_info(self,
                               obs: Observation,
                               target: Target,
                               timing_windows: List[Time]) -> TargetInfoNightIndexMap:
        """
        For a given site, calculate the information for a target for all the nights in
        the time grid and store this in the _target_information.

        Some of this information may be repetitive as, e.g. the RA and dec of a target should not
        depend on the site, so sites whose twilights overlap with have this information repeated.

        Finally, this method can calculate the total amount of time that, for the observation,
        the target is visible, and the visibility fraction for the target as a ratio of the amount of
        time remaining for the observation to the total visibility time for the target from a night through
        to the end of the period.
        """
        # Get the night events.
        night_events = self.night_events[obs.site]

        # Iterate over the time grid, checking to see if there is already a TargetInfo
        # for the target for the given day at the given site.
        # If so, we skip.
        # If not, we execute the calculations and store.
        # In order to properly calculate the:
        # * rem_visibility_time: total time a target is visible from the current night to the end of the period
        # * rem_visibility_frac: fraction of remaining observation length to rem_visibility_time
        # we want to process the nights BACKWARDS so that we can sum up the visibility time.
        rem_visibility_time = 0.0 * u.h
        rem_visibility_frac_numerator = obs.exec_time() - obs.total_used()

        target_info: TargetInfoNightIndexMap = {}

        for ridx, jday in enumerate(reversed(self.time_grid)):
            # Convert to the actual time grid index.
            night_idx = len(self.time_grid) - ridx - 1

            # Calculate the ra and dec for each target.
            # In case we decide to go with numpy arrays instead of SkyCoord,
            # this information is already stored in decimal degrees at this point.
            if isinstance(target, SiderealTarget):
                # Take proper motion into account over the time slots.
                coord = Collector._calculate_proper_motion(target, self.time_grid[night_idx])
            elif isinstance(target, NonsiderealTarget):
                coord = SkyCoord(target.ra * u.deg, target.dec * u.deg)

            else:
                msg = f'Invalid target: {target}'
                raise ValueError(msg)

            # Calculate the hour angle, altitude, azimuth, parallactic angle, and airmass.
            lst = night_events.local_sidereal_times[night_idx]
            hourangle = lst - coord.ra
            hourangle.wrap_at(12.0 * u.hour, inplace=True)
            alt, az, par_ang = sky.Altitude.above(coord.dec, hourangle, obs.site.location.lat)
            airmass = sky.true_airmass(alt)

            # Calculate the time slots for the night in which there is visibility.
            visibility_slot_idx = np.array([], dtype=int)

            # Determine time slot indices where the sky brightness and elevation constraints are met.
            # By default, in the case where an observation has no constraints, we use SB ANY.
            if obs.constraints and obs.constraints.conditions.sb < SkyBackground.SBANY:
                targ_sb = obs.constraints.conditions.sb
                targ_moon_ang = coord.separation(night_events.moon_pos[night_idx])
                brightness = sky.brightness.calculate_sky_brightness(
                    180.0 * u.deg - night_events.sun_moon_ang[night_idx],
                    targ_moon_ang,
                    night_events.moon_dist[night_idx],
                    90.0 * u.deg - night_events.moon_alt[night_idx],
                    90.0 * u.deg - alt,
                    90.0 * u.deg - night_events.sun_alt[night_idx]
                )
                sb = sky.brightness.convert_to_sky_background(brightness)
            else:
                targ_sb = SkyBackground.SBANY
                sb = np.full([len(night_events.times[night_idx])], SkyBackground.SBANY)

            # In the case where an observation has no constraint information or an elevation constraint
            # type of None, we use airmass default values.
            if obs.constraints and obs.constraints.elevation_type != ElevationType.NONE:
                targ_prop = hourangle if obs.constraints.elevation_type is ElevationType.HOUR_ANGLE else airmass
                elev_min = obs.constraints.elevation_min
                elev_max = obs.constraints.elevation_max
            else:
                targ_prop = airmass
                elev_min = Constraints.DEFAULT_AIRMASS_ELEVATION_MIN
                elev_max = Constraints.DEFAULT_AIRMASS_ELEVATION_MAX

            # Calculate the time slot indices for the night where:
            # 1. The sun altitude requirement is met (precalculated in night_events)
            # 2. The sky background constraint is met
            # 3. The elevation constraints are met
            # TODO: Are we calculating this correctly? I am not convinced.
            sa_idx = night_events.sun_alt_indices[night_idx]
            c_idx = np.where(
                np.logical_and(sb[sa_idx] <= targ_sb,
                               np.logical_and(targ_prop[sa_idx] >= elev_min,
                                              targ_prop[sa_idx] <= elev_max))
            )[0]

            # Apply timing window constraints.
            # We always have at least one timing window. If one was not given, the program length will be used.
            for tw in timing_windows:
                tw_idx = np.where(
                    np.logical_and(night_events.times[night_idx][sa_idx[c_idx]] >= tw[0],
                                   night_events.times[night_idx][sa_idx[c_idx]] <= tw[1])
                )[0]
                visibility_slot_idx = np.append(visibility_slot_idx, sa_idx[c_idx[tw_idx]])

            # Create a visibility filter that has an entry for every time slot over the night,
            # with 0 if the target is not visible and 1 if it is visible.
            visibility_slot_filter = np.zeros(len(night_events.times[night_idx]))
            visibility_slot_filter.put(visibility_slot_idx, 1.0)

            # TODO: Guide star availability for moving targets and parallactic angle modes.

            # Calculate the visibility time, the ongoing summed remaining visibility time, and
            # the remaining visibility fraction.
            # If the denominator for the visibility fraction is 0, use a value of 0.
            visibility_time = len(visibility_slot_idx) * self.time_slot_length
            rem_visibility_time += visibility_time
            if rem_visibility_time.value:
                # This is a fraction, so convert to seconds to cancel the units out.
                rem_visibility_frac = (rem_visibility_frac_numerator.total_seconds() /
                                       rem_visibility_time.to_value(u.s))
            else:
                rem_visibility_frac = 0.0

            target_info[NightIndex(night_idx)] = TargetInfo(
                coord=coord,
                alt=alt,
                az=az,
                par_ang=par_ang,
                hourangle=hourangle,
                airmass=airmass,
                sky_brightness=sb,
                visibility_slot_idx=visibility_slot_idx,
                visibility_slot_filter=visibility_slot_filter,
                visibility_time=visibility_time,
                rem_visibility_time=rem_visibility_time,
                rem_visibility_frac=rem_visibility_frac
            )

        # Return all the target info for the base target in the Observation across the nights of interest.
        return target_info

    def load_programs(self, program_provider_class: Type[ProgramProvider], data: Iterable[dict]) -> None:
        """
        Load the programs provided as JSON into the Collector.

        The program_provider should be a concrete implementation of the API to read in
        programs from JSON files.

        The json_data comprises the program inputs as an iterable object per site. We use iterable
        since the amount of data here might be enormous, and we do not want to store it all
        in memory at once.

        As this is OCS-specific, in a Program, all observations are guaranteed to be at the same site;
        however, since this may not always be the case and will not in GPP, we still process all programs
        and simply omit observations that are not at a site listed in the desired sites.
        """
        if not (isclass(program_provider_class) and issubclass(program_provider_class, ProgramProvider)):
            raise ValueError('Collector load_programs requires a ProgramProvider class as the second argument')
        program_provider = program_provider_class(self.obs_classes, self.sources)

        # Purge the old programs and observations.
        Collector._programs = {}

        # Read in the programs.
        # Count the number of parse failures.
        bad_program_count = 0

        for json_program in data:
            try:
                if len(json_program.keys()) != 1:
                    msg = f'JSON programs should only have one top-level key: {" ".join(json_program.keys())}'
                    raise ValueError(msg)

                # Extract the data from the JSON program. We do not need the top label.
                data = next(iter(json_program.values()))
                program = program_provider.parse_program(data)

                # If program could not be parsed, skip. This happens in one of three cases:
                # 1. Program semester cannot be determined from ID.
                # 2. Program type cannot be determined from ID.
                # 3. Program root group is empty.
                if program is None:
                    continue

                # If program semester is not in the list of specified semesters, skip.
                if program.semester is None or program.semester not in self.semesters:
                    logger.warning(f'Program {program.id} has semester {program.semester} (not included). Skipping.')
                    continue

                # If a program ID is repeated, warn and overwrite.
                if program.id in Collector._programs.keys():
                    logger.warning(f'Data contains a repeated program with id {program.id} (overwriting).')

                Collector._programs[program.id] = program

                # TODO HACK: Zero out times for Bryan.
                # ValidationMode._clear_observation_info(program.observations()) # noqa

                # Set the observation IDs for this program.
                # Collector._observations_per_program[program.id] = frozenset(obs.id for obs in good_obs)
                Collector._observations_per_program[program.id] = frozenset(obs.id for obs in program.observations())

                for obs in program.observations():
                    if obs.site in self.sites:
                        # Retrieve tne base target, if any. If not, we cannot process.
                        base = obs.base_target()

                        # Record the observation and target for this observation ID.
                        Collector._observations[obs.id] = obs, base

                        # This should never happen since we set an empty target for observations without a base.
                        if base is None:
                            raise RuntimeError(f'No base target found for observation {obs.id}.')

                        # Compute the timing window expansion for the observation and then calculate the target
                        # information.
                        tw = self._process_timing_windows(program, obs)
                        ti = self._calculate_target_info(obs, base, tw)
                        logger.info(f'Processed observation {obs.id}.')

                        # Compute the TargetInfo.
                        Collector._target_info[(base.name, obs.id)] = ti

            except ValueError as e:
                bad_program_count += 1
                logger.warning(f'Could not parse program: {e}')

        if bad_program_count:
            logger.error(f'Could not parse {bad_program_count} programs.')

    def night_configurations(self,
                             site: Site,
                             night_indices: NightIndices) -> Dict[NightIndices, NightConfiguration]:
        """
        Return the list of NightConfiguration for the site and nights under configuration.
        """
        return {night_idx: Collector._resource_service.get_night_configuration(
            site,
            self.get_night_events(site).time_grid[night_idx].datetime.date() - Collector._DAY
        ) for night_idx in night_indices}

    def _get_group(self, obs: Observation) -> Group:
        """Return the group that an observation is a member of."""
        # TODO: How do we handle nested scheduling groups? Right now, if in a subgroup of a scheduling group, will fail.
        program = self.get_program(obs.belongs_to)
        # print(program.id)

        # Look for obs in the specified group. Compare by ID to avoid comparing full objects.
        def find_obs(g: Group) -> bool:
            return any(obs.unique_id == group_obs.unique_id for group_obs in g.observations())

        for group in program.root_group.children:
            if group.is_scheduling_group():
                for subgroup in group.children:
                    if find_obs(subgroup):
                        return group
            else:
                if find_obs(group):
                    return group

        # This should never happen: cannot find observation in program.
        raise RuntimeError(f'Could not find observation {obs.id.id} in program {program.id.id}.')

    def time_accounting(self,
                        plans: Plans,
                        sites: FrozenSet[Site] = ALL_SITES,
                        end_timeslot_bounds: Optional[Dict[Site, Optional[TimeslotIndex]]] = None) -> None:
        """
        For the given plans, which contain a set of plans for all sites for one night,
        perform time accounting on the plans for the specified sites up until the specified
        end timeslot for the site.

        If the end timeslot bound occurs during a visit, charge up to that timeslot
        For now, scheduling groups are charged only if they can be done completely.

        If end_timeslot_idx is not specified or not specified for a given site,
        then we perform time accounting across the entire night.
        """
        # Avoids repeated conversions in loop.
        time_slot_length = self.time_slot_length.to_datetime()

        for plan in plans:
            if plan.site not in sites:
                continue

            # Determine the end timeslot for the site if one is specified.
            # We set to None is the whole night is to be done.
            end_timeslot_bound = end_timeslot_bounds.get(plan.site) if end_timeslot_bounds is not None else None

            # print(f'\ntime_accounting')
            # print(f'{plan.site} {end_timeslot_bound}')

            grpvisits = []
            # Restore this if we actually need ii, but seems it was just being used to check that grpvisits nonempty.
            # for ii, visit in enumerate(sorted(plan.visits, key=lambda v: v.start_time_slot)):
            for visit in sorted(plan.visits, key=lambda v: v.start_time_slot):
                obs = self.get_observation(visit.obs_id)
                group = self._get_group(obs)
                # print(f'{ii} {visit.obs_id.id} {group.unique_id.id} {visit.start_time_slot} '
                #       f'{visit.start_time_slot + visit.time_slots - 1}')
                if grpvisits and group.is_scheduling_group() and group == grpvisits[-1].group:
                    grpvisits[-1].visits.append(visit)
                else:
                    grpvisits.append(GroupVisits(group=group, visits=[visit]))

            for grpvisit in grpvisits:
                # print(grpvisit.group.unique_id.id, grpvisit.start_time_slot(), grpvisit.end_time_slot())
                # Determine if group should be charged
                if grpvisit.group.is_scheduling_group():
                    # For now, only change a scheduling group if it can be done fully
                    charge_group = end_timeslot_bound is None or end_timeslot_bound > grpvisit.end_time_slot()
                else:
                    charge_group = end_timeslot_bound is None or end_timeslot_bound > grpvisit.start_time_slot()

                # Charge if the end slot is less than this
                if end_timeslot_bound is not None:
                    end_timeslot_charge = end_timeslot_bound
                else:
                    end_timeslot_charge = grpvisit.end_time_slot() + 1

                # Charge to not_charged if the bound occurs during an AND (scheduling) group
                # TODO: for NIR + telluric, check if the standard was taken before the event, if so then charge for
                # what was observed and make a new copy of the telluric
                not_charged = (grpvisit.group.is_scheduling_group() and
                               grpvisit.start_time_slot() <= end_timeslot_charge <= grpvisit.end_time_slot())
                # print(f'charge_group = {charge_group}, charge_unused = {not_charged}')

                # print(f'\tGroup observations')
                # prog_obs = grpvisit.group.program_observations()
                part_obs = grpvisit.group.partner_observations()
                # print(f'\t\t Science')
                # for obs in prog_obs:
                #     print(f'\t\t {obs.unique_id.id}')
                # print(f'\t\t Partner')
                # for obs in part_obs:
                #     print(f'\t\t {obs.unique_id.id}')

                # print(f'\tVisits scheduled')
                for visit in grpvisit.visits:
                    # print(
                    #     f'\t\t{visit.obs_id.id} {visit.atom_start_idx} {visit.atom_end_idx} {visit.start_time_slot} '
                    #     f'{visit.time_slots} {visit.start_time_slot + visit.time_slots - 1}')

                    # Observation information
                    observation = self.get_observation(visit.obs_id)

                    # Number of slots in acquisition
                    n_slots_acq = time2slots(time_slot_length, observation.acq_overhead)
                    # print(f'\t\t{observation.acq_overhead.total_seconds()} {n_slots_acq}')

                    # Cumulative exec_times of unobserved atoms
                    cumul_seq = observation.cumulative_exec_times()
                    obs_seq = observation.sequence

                    # Check if the Observation has been completely observed.
                    if charge_group and visit.atom_end_idx == len(obs_seq) - 1:
                        logger.info(f'Marking observation complete: {observation.id.id}')
                        observation.status = ObservationStatus.OBSERVED
                        if observation in part_obs:
                            part_obs.remove(observation)
                    elif not_charged:
                        observation.status = ObservationStatus.ONGOING

                    # Loop over atoms
                    for atom_idx in range(visit.atom_start_idx, visit.atom_end_idx + 1):
                        # calculate end time slot for each atom and compare with end_timeslot_charge
                        slot_length_visit = n_slots_acq + time2slots(time_slot_length, cumul_seq[atom_idx])  # noqa
                        slot_atom_end = visit.start_time_slot + slot_length_visit - 1

                        # Not currently in use.
                        # if atom_idx == visit.atom_start_idx:
                        #     slot_atom_length = slot_length_visit
                        # else:
                        #     time_slots = Plan.time2slots(time_slot_length, cumul_seq[atom_idx-1])  # noqa
                        #     slot_atom_length = slot_length_visit - n_slots_acq - time_slots
                        # if slot_atom_length > 0:
                        #     slot_atom_start = slot_atom_end - slot_atom_length + 1
                        # else:
                        #     slot_atom_start = slot_atom_end - slot_atom_length

                        if slot_atom_end < end_timeslot_charge:
                            if charge_group:
                                # Charge to program or partner
                                obs_seq[atom_idx].program_used = obs_seq[atom_idx].prog_time
                                obs_seq[atom_idx].partner_used = obs_seq[atom_idx].part_time

                                # Charge acquisition to the first atom.
                                if atom_idx == visit.atom_start_idx:
                                    if observation.obs_class == ObservationClass.PARTNERCAL:
                                        obs_seq[atom_idx].program_used += observation.acq_overhead
                                    elif (observation.obs_class == ObservationClass.SCIENCE or
                                          observation.obs_class == ObservationClass.PROGCAL):
                                        obs_seq[atom_idx].program_used += observation.acq_overhead

                                obs_seq[atom_idx].observed = True
                                obs_seq[atom_idx].qa_state = QAState.PASS

                            elif not_charged:
                                # charge to not_charged
                                # obs_seq[atom_idx].not_charged += (end_timeslot_charge - slot_atom_start + 1) * self.time_slot_length
                                continue

                        # print(f'\t\t\t{observation.sequence[atom_idx].id:3} {slot_atom_start:3} {observation.sequence[atom_idx].exec_time} '
                        #     f'{cumul_seq[atom_idx]} {slot_atom_length:3} {slot_atom_end:3} observed:{obs_seq[atom_idx].observed} '
                        #     f'not_charged:{not_charged}')

                # If charging the groups, set remaining partner cals to INACTIVE
                if charge_group:
                    for obs in part_obs:
                        # print(f'\t Setting {obs.unique_id.id} to INACTIVE.')
                        logger.info(f'\tTime_accounting setting {obs.unique_id.id} to INACTIVE.')
                        obs.status = ObservationStatus.INACTIVE
