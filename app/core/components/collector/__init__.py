# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, Dict, FrozenSet, Iterable, List, NoReturn, Optional, Tuple, final

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from lucupy import sky
from lucupy.minimodel import (Constraints, ElevationType, NightIndex, NonsiderealTarget, Observation, ObservationID,
                              ObservationClass, ObservationStatus, Program, ProgramID, ProgramTypes, Resource, Semester,
                              SiderealTarget, Site, SkyBackground, Target)
from more_itertools import partition

from app.core.calculations import NightEvents, TargetInfo, TargetInfoMap, TargetInfoNightIndexMap
from app.core.components.base import SchedulerComponent
from app.core.components.nighteventsmanager import NightEventsManager
from app.core.programprovider.abstract import ProgramProvider
from mock.resource import ResourceMock


@final
@dataclass
class Collector(SchedulerComponent):
    """
    At this point, we still work with AstroPy Time for efficiency.
    We will switch do datetime and timedelta by the end of the Collector
    so that the Scheduler relies on regular Python datetime and timedelta
    objects instead.
    """
    start_time: Time
    end_time: Time
    time_slot_length: TimeDelta
    sites: FrozenSet[Site]
    semesters: FrozenSet[Semester]
    program_types: FrozenSet[ProgramTypes]
    obs_classes: FrozenSet[ObservationClass]

    # Manage the NightEvents with a NightEventsManager to avoid unnecessary recalculations.
    _night_events_manager: ClassVar[NightEventsManager] = NightEventsManager()

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
        if not np.isscalar(self.start_time.value):
            msg = f'Illegal start time (must be scalar): {self.start_time}.'
            raise ValueError(msg)
        if not np.isscalar(self.end_time.value):
            msg = f'Illegal end time (must be scalar): {self.end_time}.'
            raise ValueError(msg)
        if self.start_time >= self.end_time:
            msg = f'Start time ({self.start_time}) must be earlier than end time ({self.end_time}).'
            raise ValueError(msg)

        # Set up the time grid for the period under consideration: this is an astropy Time
        # object from start_time to end_time inclusive, with one entry per day.
        # Note that the format is in jdate.
        self.time_grid = Time(np.arange(self.start_time.jd, self.end_time.jd + 1.0, (1.0 * u.day).value), format='jd')

        # Create the night events, which contain the data for all given nights by site.
        # This may retrigger a calculation of the night events for one or more sites.
        self.night_events = {
            site: Collector._night_events_manager.get_night_events(self.time_grid, self.time_slot_length, site)
            for site in self.sites
        }

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
        return Collector._observations.values()

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
        if info is None or info[1] is None:
            return None
        target_name = info[1].name
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

            target_info[night_idx] = TargetInfo(
                coord=coord,
                alt=alt,
                az=az,
                par_ang=par_ang,
                hourangle=hourangle,
                airmass=airmass,
                sky_brightness=sb,
                visibility_slot_idx=visibility_slot_idx,
                visibility_time=visibility_time,
                rem_visibility_time=rem_visibility_time,
                rem_visibility_frac=rem_visibility_frac
            )

        # Return all the target info for the base target in the Observation across the nights of interest.
        return target_info

    def load_programs(self, program_provider: ProgramProvider, data: Iterable[dict]) -> NoReturn:
        """
        Load the programs provided as JSON into the Collector.

        The program_provider should be a concrete implementation of the API to read in
        programs from JSON files.

        The json_data comprises the program inputs as an iterable object per site. We use iterable
        since the amount of data here might be enormous, and we do not want to store it all
        in memory at once.
        """
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

                # If program not in specified semester, then skip.
                if program.semester is None or program.semester not in self.semesters:
                    logging.warning(f'Program {program.id} not in a specified semester (skipping): {program.semester}.')
                    continue

                # If a program ID is repeated, warn and overwrite.
                if program.id in Collector._programs.keys():
                    logging.warning(f'Data contains a repeated program with id {program.id} (overwriting).')
                Collector._programs[program.id] = program

                # Collect the observations in the program and sort them by site.
                # Filter out here any observation classes that have not been specified to the Collector.
                bad_obs, good_obs = partition(lambda x: x.obs_class in self.obs_classes, program.observations())
                bad_obs = list(bad_obs)
                good_obs = list(good_obs)

                for obs in bad_obs:
                    name = obs.obs_class.name
                    logging.warning(f'Observation {obs.id} not in a specified class (skipping): {name}.')

                # Set the observation IDs for this program.
                Collector._observations_per_program[program.id] = frozenset(obs.id for obs in good_obs)

                for obs in good_obs:

                    # Retrieve tne base target, if any. If not, we cannot process.
                    base = obs.base_target()

                    # Record the observation and target for this observation ID.
                    Collector._observations[obs.id] = obs, base

                    if base is None:
                        logging.warning(f'No base target found for observation {obs.id} (skipping).')
                        continue

                    # Compute the timing window expansion for the observation and then calculate the target information.
                    tw = self._process_timing_windows(program, obs)
                    ti = self._calculate_target_info(obs, base, tw)
                    logging.info(f'Processed observation {obs.id}.')

                    # Compute the TargetInfo.
                    Collector._target_info[(base.name, obs.id)] = ti

            except ValueError as e:
                bad_program_count += 1
                logging.warning(f'Could not parse program: {e}')

        if bad_program_count:
            logging.error(f'Could not parse {bad_program_count} programs.')

    def available_resources(self,
                            site: Site,
                            night_indices: npt.NDArray[NightIndex]) -> List[FrozenSet[Resource]]:
        """
        Return a set of available resources for the night under consideration.
        """
        # ResourceMock works with dates and not night_idx, so we need to convert.
        return [ResourceMock().get_resources(site, self.get_night_events(site).time_grid[night_idx].datetime.date())
                for night_idx in night_indices]
