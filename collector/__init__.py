from astropy.coordinates import Angle, SkyCoord
from astropy.units.quantity import Quantity
from astropy import units as u
import numpy as np
import time
from typing import Dict, FrozenSet, Iterable, NoReturn

from api.abstract import ProgramProvider
from common import sky_brightness
import common.helpers as helpers
from common.minimodel import *
from common.scheduler import SchedulerComponent
import common.vskyutil as vskyutil

# The package marshmallow-dataclass extends the dataclasses.dataclass decorator by adding schema information.
# We need this as a PyCharm warning workaround for dataclasses here.
# Otherwise, we get that dataclass parameters are unexpected.
from marshmallow_dataclass import add_schema


# Type aliases for convenience.
NightIndex = int


# NOTE: this is an unfortunate workaround needed to get rid of warnings in PyCharm.
@add_schema
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
    sun_moon_ang: Angle
    moon_illumination_fraction: npt.NDArray[float]

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

        # Most general times: can convert into others.
        # TODO: This does not work as currently done:
        # starts = Time(['2000-01-01', '2000-01-02'])
        # ends = Time(['2000-01-01 12:00', '2000-01-02 12:00'])
        # n = [3, 4]
        # ts = [np.linspace(start.jd, end.jd, i) for start, end, i in zip(starts, ends, n)]
        # -> [array([2451544.5 , 2451544.75, 2451545]), array([2451545.5, 2451545.66666667, 2451545.83333333, 2451546])]
        # Time(ts, format='jd') generates exceptions.
        # We can do:
        # self.times = [Time(t, format='jd') for t in ts]
        # This will give us arrays per night of the times.
        # We can then wrap all of this in a Time object to get an array, but it seems
        # more advantageous to keep them separate so we can do nightly lookups?
        # self.times = Time(self.times)
        # -> <Time object: scale='utc' format='jd' value=[array ts above in 1D]>

        # NOTE: We want this as a Python list because the entries will have different lengths!
        self.times = [Time(np.linspace(start.jd, end.jd - time_slot_length_days, i), format='jd')
                      for start, end, i in zip(time_starts, time_ends, n)]

        # Pre-calculate the different times.
        # We want these as Python lists because the entries will have different lengths.
        # TODO: Thus, the np.vectorize approach will not work here, but we keep it here as documentation.
        # self.utc_times = np.vectorize(lambda t: t.to_datetime('utc'))(self.times)
        self.utc_times = [t.to_datetime('utc') for t in self.times]
        self.local_times = [t.to_datetime(self.site.value.timezone) for t in self.times]
        self.local_sidereal_times = [vskyutil.lpsidereal(t, self.site.value.location) for t in self.times]

        def altazparang(pos: SkyCoord):
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
            return np.array(alt), np.array(az), np.array(par_ang)

        # Calculate the parameters for the sun, joining together the positions.
        self.sun_pos = SkyCoord([vskyutil.lpsun(t) for t in self.times])
        self.sun_alt, self.sun_az, self.sun_par_ang = altazparang(self.sun_pos)

        # TODO: Do we need both pos and dist? Depends on which sb algorithm we are using.
        # TODO: lpmoon is faster than accumoon so it is preferable, and just produces a SkyCoord.
        # TODO: If we need dist as well, we must use accumoon instead.
        # self.moon_pos = [vskyutil.lpmoon(t, self.site.value.location) for t in self.times]

        # accumoon produces a tuple, (SkyCoord, ndarray) indicating position and distance.
        # In order to populate both moon_pos and moon_dist, we use the zip(*...) technique to
        # collect the SkyCoords into one tuple, and the ndarrays into another.
        moon_pos, moon_dist = zip(*[vskyutil.accumoon(t, self.site.value.location) for t in self.times])
        self.moon_pos = SkyCoord(moon_pos)
        self.moon_dist = Quantity(moon_dist)
        self.moon_alt, self.moon_az, self.moon_par_ang = altazparang(self.moon_pos)


class NightEventsManager:
    """
    Manages pre-calculation of NightEvents.
    We only maintain one set of NightEvents for a site at any given time.
    """
    _night_events: ClassVar[Mapping[Site, NightEvents]]

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
                 (time_grid[0] < ne[site].time_grid[0] or time_grid[-1] > ne[site].time_grid[1]))):
            # TODO: I am not convinced that this is the correct way to calculate the night events.
            # TODO: This is how it is done in the old collector, so it should work? Needs more testing.
            NightEventsManager._night_events[site] = NightEvents(
                time_grid=time_grid,
                time_slot_length=time_slot_length,
                *vskyutil.nightevents(time_grid, site.value.location, site.value.timezone, verbose=False)
            )

        return NightEventsManager._night_events[site]


@add_schema
@dataclass
class TargetInfo:
    """
    Target information for a given target at a given site for a given night.

    For a SiderealTarget, we have to account for proper motion, which is handled below.

    For a NonsiderealTarget, we have to account for ephemeris data, which is handled in
    the mini-model and is simply brought over by reference.

    All the values here are numpy arrays for each time step at the site for the night.

    # TODO: Unsure what ivis is supposed to mean.
    """
    coord: SkyCoord
    alt: Angle
    az: Angle
    par_ang: Angle
    hourangle: Angle
    airmass: npt.NDArray[float]
    sky_brightness: npt.NDArray[SkyBackground]
    ivis: npt.NDArray[int]

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

    # The observations associated with the above programs, which we store by site.
    _observations: ClassVar[Dict[Site, List[Observation]]] = {}

    # Observation timing windows in an easier to use format.
    # We look up timing window information by observation id.
    _timing_windows: ClassVar[Dict[ObservationID, List[Time]]]

    # The target information is dependent on the:
    # 1. Target
    # 2. ObservationID (for the associated constraints and site)
    # 4. NightIndex of interest
    _target_info: ClassVar[Dict[(Target, ObservationID, NightIndex)], TargetInfo] = {}

    # The default timeslot length currently used.
    DEFAULT_TIMESLOT_LENGTH: ClassVar[Time] = 1.0 * u.min

    # These are exclusive to the create_time_array.
    _MIN_NIGHT_EVENT_TIME: ClassVar[Time] = Time('1980-01-01 00:00:00', format='iso', scale='utc')

    # NOTE: This logs an ErfaWarning about dubious year. This is due to using a future date and not knowing
    # how many leap seconds have happened: https://github.com/astropy/astropy/issues/5809
    _MAX_NIGHT_EVENT_TIME: ClassVar[Time] = Time('2100-01-01 00:00:00', format='iso', scale='utc')

    # The number of milliarcsecs in a degree, for proper motion calculation.
    _MILLIARCSECS_PER_DEGREE: ClassVar[int] = 1296000000

    # Information for Julian years.
    _JULIAN_BASIS: ClassVar[float] = 2451545.0
    _JULIAN_YEAR_LENGTH: ClassVar[float] = 365.25

    def __post_init__(self):
        """
        Initializes the internal data structures for the Collector and populates them.
        """
        # TODO: How to handle the time range?
        # TODO: 1. Originally: Time, assume size 2
        # TODO: 2. start Time and end Time, force size 1
        # Check that the times are valid.
        if self.start_time >= self.end_time:
            msg = f'Start time ({self.start_time}) must be earlier than end time ({self.end_time}).'
            logging.error(msg)
            raise ValueError(msg)

        # Set up the time grid for the period under consideration: this is an astropy Time
        # object from start_time to end_time inclusive, with one entry per day.
        # Note that the format is in jdate.
        self.time_grid = Time(np.arange(self.start_time.jd, self.end_time.jd + 1.0, (1.0 * u.day).value), format='jd')

        # Create the night events, which contain the data for all given nights by site.
        # This may retrigger a calculation of the night events for one or more sites.
        self.night_events = {
            site: self._night_events_manager.get_night_events(self.time_grid, self.time_slot_length, site)
            for site in self.sites
        }

    def _process_timing_windows(self, prog: Program, obs: Observation) -> NoReturn:
        """
        Given an Observation, convert the TimingWindow information in it to a simpler format
        to verify by converting each TimingWindow representation to a collection of Time frames
        based on the start, duration, repeat, and period.

        If no timing windows are given, then create one large timing window for the entire program.

        TODO: It would be easy to convert this from astropy to simply use datetime.
        TODO: How would this affect efficiency?
        """
        if len(obs.constraints.timing_windows) == 0:
            # Create a timing window for the entirety of the program.
            windows = [Time([prog.start, prog.end])]
        else:
            windows = []
            for tw in obs.constraints.timing_windows:
                t0 = time.mktime(tw.start.utctimetuple()) * 1000 * u.ms
                begin = Time(t0.to_value('s'), format='unix', scale='utc')
                duration = tw.duration / 3600000.0 * u.h
                repeat = max(1, tw.repeat)
                period = tw.period.total_seconds() / 3600.0 * u.h if tw.period is not None else 0.0 * u.h
                windows.extend([Time([begin + i * period, begin + i * period + duration]) for i in range(repeat)])

        self._timing_windows[obs.id] = windows

    def _calculate_target_info(self,
                               obs: Observation,
                               target: Target,
                               use_sb2: bool = True,
                               extras: bool = True) -> NoReturn:
        """
        For a given site, calculate the information for a target for all the nights in
        the time grid and store this in the _target_information.

        Some of this information may be repetitive as, e.g. the RA and dec of a target should not
        depend on the site, so sites whose twilights overlap with have this information repeated.
        """
        night_events = Collector._night_events_manager.get_night_events(self.time_grid, self.time_slot_length, obs.site)

        # Iterate over the time grid, checking to see if there is already a TargetInfo
        # for the target for the given day at the given site.
        # If so, we skip.
        # If not, we execute the calculations and store.
        for idx, jday in enumerate(self.time_grid):
            if (target, obs.id, idx) not in Collector._target_info:
                if isinstance(target, SiderealTarget):
                    pm_ra = target.pm_ra / Collector._MILLIARCSECS_PER_DEGREE
                    pm_dec = target.pm_dec / Collector._MILLIARCSECS_PER_DEGREE

                    # Calculate the new coordinates for the night.
                    # For each entry in time, we want to calculate the offset in epoch-years.
                    time_offsets = np.array([target.epoch + (t - Collector._JULIAN_BASIS) / Collector._JULIAN_YEAR_LENGTH
                                             for t in night_events.times[idx]])

                    # Calculate the ra and dec for each target and convert to decimal degrees.
                    # TODO: Verify, but we should already be in decimal degrees here.
                    # TODO: Leaving this in for now as documentation on SkyCoord should it be needed.
                    coords = SkyCoord((target.ra + pm_ra * time_offsets) * u.deg,
                                      (target.dec + pm_dec * time_offsets) * u.deg)
                    # ra = coords.ra.value
                    # dec = coords.dec.value
                    # ra = target.ra + pm_ra * time_offsets
                    # dec = target.dec + pm_dec * time_offsets

                elif isinstance(target, NonsiderealTarget):
                    coords = SkyCoord(target.ra * u.deg, target.dec * u.deg)
                    # ra = target.ra
                    # dec = target.dec

                else:
                    msg = f'Invalid target: {target}'
                    logging.error(msg)
                    raise ValueError(msg)

                # Calculate the hour angle, altitude, azimuth, parallactic angle, and airmass.
                lst = night_events.local_sidereal_times[idx]
                hourangle = lst - coords.ra
                hourangle.wrap_at(12.0 * u.hour, inplace=True)
                alt, az, par_ang = vskyutil.altazparang(coords.dec, hourangle, obs.site.value.location.lat)
                airmass = vskyutil.true_airmass(alt)

                targ_prop = airmass if obs.constraints.elevation_type is ElevationType.AIRMASS else hourangle.value

                # Sky brightness.
                sb = np.full([len(night_events.times)], SkyBackground.SBANY)

                if obs.constraints.conditions.sb < SkyBackground.SBANY:
                    targ_moon_ang = coords.separation(night_events.moon_pos)

                    # Use the new sky brightness calculator.
                    if use_sb2:
                        brightness = sky_brightness.calculate_sky_brightness(
                            180.0 * u.deg - night_events.sun_moon_ang,
                            targ_moon_ang,
                            night_events.moon_dist,
                            90.0 * u.deg - night_events.moon_alt,
                            90.0 * u.deg - alt,
                            90.0 * u.deg - night_events.sun_alt
                        )

                    # Use the QPT sky brightness calculator.
                    else:
                        brightness = sky_brightness.calculate_sky_brightness_qpt(
                            180.0 * u.deg - night_events.sun_moon_ang,
                            targ_moon_ang,
                            90.0 * u.deg - night_events.moon_alt,
                            90.0 * u.deg - alt,
                            90.0 * u.deg - night_events.sun_alt
                        )

                    sb = sky_brightness.convert_to_sky_background(brightness)

                # TODO: ivis... what is this, exactly? Not used in old selector code.
                ivis = np.array([], dtype=int)

                # Select where sky brightness and elevation constraints are met.
                # TODO: Unsure of why we select index 0 here. Is it first index where met?
                isb = np.where(np.logical_and(sb <= obs.constraints.conditions.sb,
                                              np.logical_and(night_events.sun_alt <= -12 * u.deg,
                                                             np.logical_and(
                                                                 targ_prop >= obs.constraints.elevation_min,
                                                                 targ_prop <= obs.constraints.elevation_max
                                                             ))))[0]

                # Apply timing window constraints.
                for timing in self._timing_windows[obs.id]:
                    itw = np.where(np.logical_and(night_events.times[isb] >= timing[0],
                                                  night_events.times[isb] <= timing[1]))[0]
                    ivis = np.append(ivis, isb[itw])

                # TODO: Guide star availability for moving targets and parallactic angle modes.
                # TODO: Unsure of how to calculate this.

                Collector._target_info[(target, obs.id, idx)] = TargetInfo(
                    coords=coords,
                    alt=alt,
                    az=az,
                    par_ang=par_ang,
                    hourangle=hourangle,
                    airmass=airmass,
                    sky_brightness=sb,
                    ivis=ivis
                )

    # TODO: This should also be precomputed: it is not clear how it will change if
    # TODO: we get a mutation notification from GPP, which may happen with execution time or
    # TODO: time allocation.
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
        self._timing_windows = {}

        # As we read, keep track of the observations per site.
        observations: Dict[Site, List[Observation]] = {site: [] for site in self.sites}

        # Read in the programs.
        # Count the number of parse failures.
        bad_program_count = 0
        for json_program in data:
            try:
                if len(json_program.keys()) != 1:
                    msg = f'JSON programs should only have one top-level key: {" ".join(json_program.keys())}'
                    logging.error(msg)
                    raise ValueError(msg)

                # Extract the data from the JSON program. We do not need the top label.
                data = next(iter(json_program.values()))
                program = program_provider.parse_program(data)

                # If program not in specified semester, then skip.
                if program.semester is None or program.semester not in self.semesters:
                    logging.warning(f'Program {program.id} not in a specified semester (skipping).')
                    continue

                # If a program ID is repeated, warn and overwrite.
                if program.id in self._programs.keys():
                    logging.warning(f'Data contains a repeated program with id {program.id} (overwriting).')
                self._programs[program.id] = program

                # Collect the observations in the program and sort them by site.
                # Filter out here any observation classes that have not been specified to the Collector.
                for obs in program.observations():
                    if obs.obs_class in self.obs_classes:
                        observations[obs.site].append(obs)
                    else:
                        logging.warning(f'Observation {obs.id} not in a specified observation class (skipping).')
                        continue

                # Process the timing window information per observation.
                for site in self.sites:
                    for obs in observations[site]:
                        self._process_timing_windows(program, obs)

                # Process the targets over observations to calculate the target information, which contains
                # visibility information that will be used in the remaining visibility calculations.
                for site in self.sites:
                    for obs in observations[site]:
                        for target in obs.targets:
                            self._calculate_target_info(obs, target)

            except ValueError as e:
                bad_program_count += 1
                logging.warning(f'Could not parse program: {e}')

        if bad_program_count:
            logging.error(f'Could not parse {bad_program_count} programs.')

        # TODO: progress bar with tqdm?
        num_nights = len(self.time_grid)
        num_observations = len(observations.items())
        obs_visibility_hours = {site: np.zeros((len(observations[site]), num_nights)) for site in self.sites}

        for site, site_observations in observations.items():
            visibility = ...

    def available_resources(self) -> Set[Resource]:
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

    def conditions(self) -> Conditions:
        return Conditions(
            CloudCover.CC50,
            ImageQuality.IQ70,
            SkyBackground.SB20,
            WaterVapor.WVANY
        )
