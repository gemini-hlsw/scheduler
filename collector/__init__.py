from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy import units as u
import numpy as np
from typing import Dict, FrozenSet, Iterable, NoReturn
from marshmallow_dataclass import add_schema

from api.abstract import ProgramProvider
import common.helpers as helpers
from common.minimodel import *
from common.scheduler import SchedulerComponent
import common.vskyutil as vskyutil


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
    sun_moon_angle: Angle
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
        self.times = [Time(np.linspace(start.jd, end.jd - time_slot_length_days, i), format='jd')
                      for start, end, i in zip(time_starts, time_ends, n)]

        # Pre-calculate the different times.
        self.utc_times = self.times.to_datetime('utc')
        self.local_times = self.times.to_datetime(self.site.value.timezone)
        self.local_sidereal_times = vskyutil.lpsidereal(self.times, self.site.value.location)

        # Create lists / dicts corresponding to the time grid with an AstroPy SkyCoord array giving
        # the sun or moon position at each time step. Can access via ra and dec members.
        # Sun / moon positions in RA / Dec and then for each site in AltAz and parallactic angle.
        # Alt-Az and parallactic angle are of type Angle.
        self.sun_position_radec = vskyutil.lpsun(self.utc_times)
        self.sun_position_alt, self.sun_position_az, self.sun_parallactic_angle = \
            vskyutil.altazparang(self.sun_position_radec.dec,
                                 self.local_sidereal_times - self.sun_position_radec.ra,
                                 self.site.value.location.lat)
        self.moon_position_radec = vskyutil.lpmoon(self.times, self.site.value.location)
        self.moon_position_alt, self.moon_position_az, self.moon_position_parallactic_angle = \
            vskyutil.altazparang(self.moon_position_radec.dec,
                                 self.local_sidereal_times - self.moon_position_radec.ra,
                                 self.site.value.location.lat)

        # Lunar distance and altitude.
        # This is a tuple (SkyCoord, distance) for the site at the given time array.
        self.moon_location_and_distance = vskyutil.accumoon(self.local_times, self.site.value.location)


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
            NightEventsManager._night_events[site] = NightEvents(
                time_grid=time_grid,
                time_slot_length=time_slot_length,
                *vskyutil.nightevents(time_grid, site.value.location, site.value.timezone, verbose=False)
            )

        return NightEventsManager._night_events[site]


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

    # This should not be populated, but we put it here instead of in __post_init__ to eliminate warnings.
    # This is a list of the programs as read in.
    # We only want to read these in once unless the program_types change.
    _programs: ClassVar[Dict[str, Program]] = {}

    # The observations associated with the above programs, which we store by site.
    _observations: ClassVar[Dict[Site, List[Observation]]] = {}

    # Calculate RA and dec with proper motion from sunset to sunrise for each day
    # a program is valid.
    # TODO: Since nights will not be the same length, we index by night. Is this right?
    # TODO: Since we calculate per site, this is not ideal as there will be overlap.
    sidereal_target_ra: ClassVar[Dict[(Site, str, int), npt.NDArray[float]]] = {}
    sidereal_target_dec: ClassVar[Dict[(Site, str, int), npt.NDArray[float]]] = {}

    # We manage the NightEvents with a NightEventsManager to avoid unnecessary
    # recalculations.
    _night_events_manager: ClassVar[NightEventsManager] = NightEventsManager()

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

    # TODO: This should also be precomputed: it is not clear how it will change if
    # TODO: we get a mutation notification.
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

        # Keep track of the sidereal targets to calculate proper motion.
        # TODO: This is not ideal as this data will be repeated for any Site night overlaps,
        # TODO: and it is not different from site to site during these overlaps.
        # TODO: How will this be handled in GPP? This may not be the place to put this.
        sidereal_targets: Mapping[Site, Set[SiderealTarget]] = {}

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
                data = list(json_program.values())[0]
                program = program_provider.parse_program(data)

                # If program not in specified semester, then stop processing.
                if program.semester is None or program.semester not in self.semesters:
                    logging.warning(f'Program {program.id} not in a specified semester (skipping).')
                    continue

                if program.id in self._programs.keys():
                    logging.warning(f'Data contains a repeated program with id {program.id} (overwriting).')
                self._programs[program.id] = program

                # Collect the observations in the program and sort them by site.
                for obs in program.observations():
                    # TODO: Confirm this is the proper place to do this.
                    # TODO: Since we are passing obs_classes to the Collector, assume it must be.
                    if obs.obs_class in self.obs_classes:
                        observations[obs.site].append(obs)
                        obs_sidereal_targets = [t for t in obs.targets if isinstance(t, SiderealTarget)]
                        sidereal_targets[obs.site].update(obs_sidereal_targets)
                    else:
                        logging.warning(f'Observation {obs.id} not in a specified class (skipping).')

            except ValueError as e:
                bad_program_count += 1
                logging.warning(f'Could not parse program: {e}')

        if bad_program_count:
            logging.error(f'Could not parse {bad_program_count} programs.')

        # Now we process the targets per site to get the necessary information.
        # Indexed by site and target name, result is an array of nights with each
        # entry an array of coordinates, either RA or Dec.

        # TODO: progress bar with tqdm?
        num_nights = len(self.time_grid)
        num_observations = len(observations.items())
        obs_visibility_hours = {site: np.zeros((len(observations[site]), num_nights)) for site in self.sites}

        for site, site_observations in observations.items():
            visibility = ...

        for site, targets in sidereal_targets.items():
            # For each time step under each night under consideration, we want to calculate:
            # 1. Coordinates using proper motion (stored as numpy array for the night)
            for target in targets:
                # Convert the proper motion to degrees.
                pm_ra = target.pm_ra / Collector._MILLIARCSECS_PER_DEGREE
                pm_dec = target.pm_dec / Collector._MILLIARCSECS_PER_DEGREE

                # TODO: This information should be precalculated and stored.
                # TODO: Also seems wrong to hard-code Epoch?
                for night in self.time_grid:
                    night_events = self.night_events[site][night]

                    sunset = helpers.round_minute(night_events.sunset, up=True)
                    sunrise = helpers.round_minute(night_events.sunrise, up=True)
                    time_slot_length_days = self.time_slot_length.to(u.day).value
                    n = np.int((sunset.jd - sunrise.jd) / time_slot_length_days + 0.5)
                    time = Time(np.linspace(sunset.jd, sunrise.jd - time_slot_length_days, n))

                    # For each entry in time, we want to calculate the offset in epoch-years.
                    time_offsets = target.epoch + (time - Collector._JULIAN_BASIS) / Collector._JULIAN_YEAR_LENGTH

                    # Calculate the ra and dec for each target.
                    target_ra = target.ra + pm_ra * time_offsets
                    target_dec = target.dec + pm_dec * time_offsets
                    target_coords = SkyCoord(target_ra * u.deg, target_dec * u.deg)

                    # Convert to Alt-Az.
                    altaz_converter = AltAz(location=site.value.location, obstime=night_events.local_times)
                    target_altaz = target_coords.transform_to(altaz_converter)

                    # Calculate the hour angles.
                    target_hourangle = vskyutil.ha_alt(target_dec,
                                                       target_coords.lat,
                                                       target_altaz.alt)

                    # Calculate the airmasses.
                    sidereal_target_airmass = vskyutil.true_airmass(target_altaz.alt)

    def available_resources(self) -> Set[Resource]:
        """
        Return a set of available resources for the period under consideration.
        """
        # TODO: Add more.
        return {
            Resource('PWFS1', 'PWFS1', None),
            Resource('PWFS2', 'PWFS2', None),
            Resource('GMOS OIWFS', 'GMOS OIWFS', None),
            Resource('GMOSN', 'GMOSN', None)
        }

    def conditions(self) -> Conditions:
        return Conditions(
            CloudCover.CC50,
            ImageQuality.IQ70,
            SkyBackground.SB20,
            WaterVapor.WVANY
        )
