import logging
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta
from astropy import units as u
from more_itertools import partition
import numpy as np
import pytz
import time
from tqdm import tqdm
from typing import Dict, FrozenSet, Iterable, Tuple, NoReturn

from api.abstract import ProgramProvider
from common import sky_brightness
import common.helpers as helpers
from common.minimodel import *
from common.scheduler import SchedulerComponent
import common.vskyutil as vskyutil

# Type aliases for convenience.
NightIndex = int


# NOTE: this is an unfortunate workaround needed to get rid of warnings in PyCharm.
# @add_schema
@dataclass
class NightEvents:
    """
    Represents night events for a given site for the period under consideration
    with the specified time slot length granularity.

    This data is maintained unless the time_grid or granularity are no longer
    compatible with the original data.
    """
    time_grid: Time
    time_slot_length: TimeDelta
    site: Site
    midnight: Time
    sunset: Time
    sunrise: Time
    twilight_evening_12: Time
    twilight_morning_12: Time
    moonrise: Time
    moonset: Time

    # Information for Julian years.
    _JULIAN_BASIS: ClassVar[float] = 2451545.0
    _JULIAN_YEAR_LENGTH: ClassVar[float] = 365.25

    def __post_init__(self):
        """
        Initialize remaining members that depend on the parameters.
        """
        # Calculate the length of each night at this site, i.e. time between twilights.
        self.night_length = (self.twilight_morning_12 - self.twilight_evening_12).to_value('h') * u.h

        # Create the time arrays, which are arrays that represent the earliest starting
        # time to the latest ending time, divided into segments of length time_slot_length.
        # We want one entry per time slot grid, i.e. per night, measured in UTC, local, and local sidereal.
        time_slot_length_days = self.time_slot_length.to(u.day).value
        time_starts = helpers.round_minute(self.twilight_evening_12, up=True)
        time_ends = helpers.round_minute(self.twilight_morning_12, up=True)

        # n in an array with the number of time slots in a night.
        n = ((time_ends.jd - time_starts.jd) / time_slot_length_days + 0.5).astype(int)

        # Calculate a list of arrays per night of the times.
        # We want this as a Python list because the arrays will have different lengths.
        self.times = [Time(np.linspace(start.jd, end.jd - time_slot_length_days, i), format='jd')
                      for start, end, i in zip(time_starts, time_ends, n)]

        # Pre-calculate the different times.
        # We want these as Python lists because the entries will have different lengths.
        self.utc_times = [t.to_datetime(pytz.UTC) for t in self.times]
        self.local_times = [t.to_datetime(self.site.value.timezone) for t in self.times]
        self.local_sidereal_times = [vskyutil.lpsidereal(t, self.site.value.location) for t in self.times]

        def altazparang(pos: List[SkyCoord]) -> Tuple[npt.NDArray[Angle], npt.NDArray[Angle], npt.NDArray[Angle]]:
            """
            Common code to invoke vskyutil.altazparang for a number of positions and then properly
            combine the results into three numpy arrays, representing, indexed by time:
            1. altitude
            2. azimuth
            3. parallactic angle
            """
            alt, az, par_ang = zip(
                *[vskyutil.altazparang(p.dec, lst - p.ra, self.site.value.location.lat)
                  for p, lst in zip(pos, self.local_sidereal_times)]
            )
            return alt, az, par_ang

        # Calculate the parameters for the sun, joining together the positions.
        # We also precalculate the indices for the time slots for the night that have the required sun altitude.
        self.sun_pos = [SkyCoord(vskyutil.lpsun(t)) for t in self.times]
        self.sun_alt, self.sun_az, self.sun_par_ang = altazparang(self.sun_pos)
        self.sun_alt_indices = [self.sun_alt[night_idx] <= -12 * u.deg for night_idx, _ in enumerate(self.time_grid)]

        # accumoon produces a tuple, (SkyCoord, ndarray) indicating position and distance.
        # In order to populate both moon_pos and moon_dist, we use the zip(*...) technique to
        # collect the SkyCoords into one tuple, and the ndarrays into another.
        # The moon_dist are already a Quantity: error if try to convert.
        self.moon_pos, self.moon_dist = zip(*[vskyutil.accumoon(t, self.site.value.location) for t in self.times])
        self.moon_alt, self.moon_az, self.moon_par_ang = altazparang(self.moon_pos)

        # Angle between the sun and the moon.
        self.sun_moon_ang = [sun_pos.separation(moon_pos) for sun_pos, moon_pos in zip(self.sun_pos, self.moon_pos)]

        # List of numpy arrays used to calculate proper motion of sidereal targets for the night.
        self.pm_array = [(t.value - NightEvents._JULIAN_BASIS) / NightEvents._JULIAN_YEAR_LENGTH for t in self.times]


class NightEventsManager:
    """
    Manages pre-calculation of NightEvents.
    We only maintain one set of NightEvents for a site at any given time.
    """
    _night_events: Dict[Site, NightEvents] = {}

    @staticmethod
    def get_night_events(time_grid: Time,
                         time_slot_length: TimeDelta,
                         site: Site) -> NightEvents:
        """
        Retrieve NightEvents. These may contain more information than requested,
        but never less.
        """
        ne = NightEventsManager._night_events

        # Recalculate if we are not compatible.
        if (site not in ne or
                time_slot_length != ne[site].time_slot_length or
                (len(ne[site].time_grid) == 1 and ne[site].time_grid[0] != time_grid[0]) or
                (len(ne[site].time_grid) > 1 and
                 (time_grid[0] < ne[site].time_grid[0] or time_grid[-1] > ne[site].time_grid[-1]))):
            # For some strange reason, this does not work if we specify keywords for NightEvents.
            # It complains about __init__() getting multiple args for time_grid.
            night_events = NightEvents(
                time_grid,
                time_slot_length,
                site,
                *vskyutil.nightevents(time_grid, site.value.location, site.value.timezone, verbose=False)
            )
            NightEventsManager._night_events[site] = night_events

        return NightEventsManager._night_events[site]


@dataclass
class TargetInfo:
    """
    Target information for a given target at a given site for a given night.

    For a SiderealTarget, we have to account for proper motion, which is handled below.

    For a NonsiderealTarget, we have to account for ephemeris data, which is handled in
    the mini-model and is simply brought over by reference.

    All the values here except:
    * visibility_time
    * rem_visibility_time
    * rem_visibility_frac
    are numpy arrays for each time step at the site for the night.

    Note that visibilities consists of the indices into the night split into time_slot_lengths
    where the necessary conditions for visibility are met, i.e.
    1. The sky brightness constraints are met.
    2. The sun altitude is below -12 degrees.
    3. The elevation constraints are met.
    4. There is an available timing window.

    visibility_time is the time_slot_length multiplied by the size of the visibilities array,
    giving the amount of time during the night that the target is visible for the observation.

    rem_visibility_time is the remaining visibility time for the target for the observation across
    the rest of the time period.
    """
    coord: SkyCoord
    alt: Angle
    az: Angle
    par_ang: Angle
    hourangle: Angle
    airmass: npt.NDArray[float]
    sky_brightness: npt.NDArray[SkyBackground]
    visibility_slot_idx: npt.NDArray[int]
    visibility_time: TimeDelta
    rem_visibility_time: TimeDelta
    rem_visibility_frac: float


# Type aliases for TargetInfo information.
# Use Dict here instead of Mapping to bypass warnings as we need [] access.
NightIndexMap = Dict[NightIndex, TargetInfo]
TargetInfoMap = Dict[Tuple[TargetName, ObservationID], NightIndexMap]


# @add_schema
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
    _programs: ClassVar[Mapping[ProgramID, Program]] = {}

    # A set of ObservationIDs per ProgramID.
    _observations_per_program: ClassVar[Mapping[ProgramID, Set[ObservationID]]] = {}

    # This is a map of observation information that is computed as the programs
    # are read in. It contains both the Observation and the base Target (if any) for
    # the observation.
    _observations: ClassVar[Mapping[ObservationID, Tuple[Observation, Optional[Target]]]] = {}

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
    _MILLIARCSECS_PER_DEGREE: ClassVar[int] = 1296000000

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

    @staticmethod
    def get_program_ids() -> Iterable[ProgramID]:
        """
        Return a list of all the program IDs stored in the Collector.
        """
        return Collector._programs.keys()

    @staticmethod
    def get_program(prog_id: ProgramID) -> Optional[Program]:
        """
        If a program with the given ID exists, return it.
        Otherwise, return None.
        """
        return Collector._programs.get(prog_id, None)

    @staticmethod
    def get_observation_ids(prog_id: Optional[ProgramID] = None) -> Optional[Iterable[ObservationID]]:
        """
        Return the observation IDs in the Collector.
        If the prog_id is specified, limit these to those in the specified in the program.
        If no such prog_id exists, return None.
        If no prog_id is specified, return a complete list of observation IDs.
        """
        if prog_id is None:
            return Collector._observations.keys()
        return Collector._observations_per_program.get(prog_id, None)

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
    def get_target_info(obs_id: ObservationID) -> Optional[TargetInfoMap]:
        """
        Given an ObservationID, if the observation exists and there is a target for the
        observation, return the target information as a map from NightIndex to TargetInfo.
        """
        info = Collector.get_observation_and_base_target(obs_id)
        if info is None or info[1] is None:
            return None
        target_name = info[1].name
        return Collector._target_info.get((obs_id, target_name), None)

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

    def _calculate_target_info(self,
                               obs: Observation,
                               target: Target,
                               timing_windows: List[Time]): # -> NightIndexMap:
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

        target_info: NightIndexMap = {}

        for ridx, jday in enumerate(reversed(self.time_grid)):
            # Convert to the actual time grid index.
            idx = len(self.time_grid) - ridx - 1

            # Calculate the ra and dec for each target.
            # In case we decide to go with numpy arrays instead of SkyCoord,
            # this information is already stored in decimal degrees at this point.
            if isinstance(target, SiderealTarget):
                # Take proper motion into account over the time slots.
                pm_ra = target.pm_ra / Collector._MILLIARCSECS_PER_DEGREE
                pm_dec = target.pm_dec / Collector._MILLIARCSECS_PER_DEGREE

                # Calculate the new coordinates for the night.
                # For each entry in time, we want to calculate the offset in epoch-years.
                # TODO: Is this right? It follows the convention in OCS Epoch.scala.
                # https://github.com/gemini-hlsw/ocs/blob/ba542ec6ffe5d03a0f31f880a52f60dd6ade3812/bundle/edu.gemini.spModel.core/src/main/scala/edu/gemini/spModel/core/Epoch.scala#L28
                time_offsets = target.epoch + night_events.pm_array[idx]
                coord = SkyCoord((target.ra + pm_ra * time_offsets) * u.deg,
                                 (target.dec + pm_dec * time_offsets) * u.deg)

            elif isinstance(target, NonsiderealTarget):
                coord = SkyCoord(target.ra * u.deg, target.dec * u.deg)

            else:
                msg = f'Invalid target: {target}'
                raise ValueError(msg)

            # Calculate the hour angle, altitude, azimuth, parallactic angle, and airmass.
            lst = night_events.local_sidereal_times[idx]
            hourangle = lst - coord.ra
            hourangle.wrap_at(12.0 * u.hour, inplace=True)
            alt, az, par_ang = vskyutil.altazparang(coord.dec, hourangle, obs.site.value.location.lat)
            airmass = vskyutil.true_airmass(alt)

            # Calculate the time slots for the night in which there is visibility.
            visibility_slot_idx = np.array([], dtype=int)

            # Determine time slot indices where the sky brightness and elevation constraints are met.
            # By default, in the case where an observation has no constraints, we use SB ANY.
            if obs.constraints and obs.constraints.conditions.sb < SkyBackground.SBANY:
                targ_sb = obs.constraints.conditions.sb
                targ_moon_ang = coord.separation(night_events.moon_pos[idx])
                brightness = sky_brightness.calculate_sky_brightness(
                    180.0 * u.deg - night_events.sun_moon_ang[idx],
                    targ_moon_ang,
                    night_events.moon_dist[idx],
                    90.0 * u.deg - night_events.moon_alt[idx],
                    90.0 * u.deg - alt,
                    90.0 * u.deg - night_events.sun_alt[idx]
                )
                sb = sky_brightness.convert_to_sky_background(brightness)
            else:
                targ_sb = SkyBackground.SBANY
                sb = np.full([len(night_events.times[idx])], SkyBackground.SBANY)

            # In the case where an observation has no constraint information or an elevation constraint
            # type of None, we use airmass default values.
            if obs.constraints:
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
            sa_idx = night_events.sun_alt_indices[idx]
            c_idx = np.where(
                np.logical_and(sb[sa_idx] <= targ_sb,
                               np.logical_and(targ_prop[sa_idx] >= elev_min,
                                              targ_prop[sa_idx] <= elev_max))
            )[0]

            # Apply timing window constraints.
            # We always have at least one timing window. If one was not given, the program length will be used.
            for tw in timing_windows:
                tw_idx = np.where(
                    np.logical_and(night_events.times[idx][c_idx] >= tw[0],
                                   night_events.times[idx][c_idx] <= tw[1])
                )[0]
                visibility_slot_idx = np.append(visibility_slot_idx, sa_idx[c_idx[tw_idx]])

            # TODO: Guide star availability for moving targets and parallactic angle modes.

            # Calculate the visibility time, the ongoing summed remaining visibility time, and
            # the remaining visibility fraction.
            # If the denominator for the visibility fraction is 0, use a value of 0.
            visibility_time = len(visibility_slot_idx) * self.time_slot_length
            rem_visibility_time += visibility_time
            if rem_visibility_time.value:
                rem_visibility_frac = rem_visibility_frac_numerator / rem_visibility_time
            else:
                rem_visibility_frac = 0.0

            target_info[idx] = TargetInfo(
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
        self._programs = {}

        # Read in the programs.
        # Count the number of parse failures.
        bad_program_count = 0

        for json_program in tqdm(data):
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
                obsvds = program.observations()
                bad_obs, good_obs = partition(lambda x: x.obs_class in self.obs_classes, program.observations())
                bad_obs = list(bad_obs)
                good_obs = list(good_obs)

                for obs in bad_obs:
                    name = obs.obs_class.name
                    logging.warning(f'Observation {obs.id} not in a specified class (skipping): {name}.')

                # Set the observation IDs for this program.
                Collector._observations_per_program[program.id] = {obs.id for obs in good_obs}

                for obs in tqdm(good_obs, leave=False):
                    # Retrieve tne base target, if any. If not, we cannot process.
                    base = next(filter(lambda t: t.type == TargetType.BASE, obs.targets), None)

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

    @staticmethod
    def available_resources() -> Set[Resource]:
        """
        Return a set of available resources for the period under consideration.
        """
        # TODO: Add more.
        return {
            Resource(id='PWFS1'),
            Resource(id='PWFS2'),
            Resource(id='GMOS OIWFS'),
            Resource(id='GMOSN')
        }

    @staticmethod
    def get_actual_conditions_variant() -> Optional[Variant]:
        time_blocks = Time(["2021-04-24 04:30:00", "2021-04-24 08:00:00"], format='iso', scale='utc')
        variants = {
            Variant(
                iq=ImageQuality.IQ70,
                cc=CloudCover.CC50,
                wv=WaterVapor.WVANY,
                wind_dir=330.0 * u.deg,
                wind_sep=40.0 * u.deg,
                wind_spd=5.0 * u.m / u.s,
                time_blocks=time_blocks
            )
        }
        return next(filter(lambda v: v.iq == ImageQuality.IQ70 and v.cc == CloudCover.CC50, variants), None)
