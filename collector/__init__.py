from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy import units as u
from collections import defaultdict
from dataclasses import field
import numpy as np
from typing import Dict, FrozenSet, Iterable, NoReturn, Tuple

from api.abstract import ProgramProvider
import common.helpers as helpers
from common.minimodel import *
from common.scheduler import SchedulerComponent
import common.vskyutil as vskyutil


@dataclass
class NightEvents:
    """
    Represents night events for the period under consideration as represented
    by the time. NightEvents should be managed by NightEventsCache for efficiency.
    """
    midnight: Time
    sunset: Time
    sunrise: Time
    twi_eve12: Time
    twi_mor12: Time
    moonrise: Time
    moonset: Time
    sun_moon_ang: Angle
    moon_illum: npt.NDArray[float]

    def __post_init__(self):
        """
        Initialize any members depending on the parameters.
        """
        self.night_length: Time = (self.twi_mor12 - self.twi_eve12).to_value('h') * u.h


@dataclass
class NightEventsCache:
    """
    A cache of NightEvents for a given night.
    We cache as these are expensive computations and we wish to avoid repeated
    calculations.
    """
    # By default, just allow the equivalent of a year's worth of data for each site.
    cache_size = 365 * len(Site)

    class CacheKeys(IntEnum):
        """
        Probably not necessary, but an index into the collection of caches in case it is needed.
        """
        MIDNIGHT = 0
        SUNSET = 1
        SUNRISE = 2
        TWI_EVE12 = 3
        TWI_MOR12 = 4
        MOONRISE = 5
        MOONSET = 6
        SUN_MOON_ANG = 7
        MOON_ILLUM = 8

    def __post_init__(self):
        """
        Initialize the caches, which are FIFO (for now).
        """
        self.caches = [{} for _ in range(len(self.CacheKeys))]
        self._fifo_entries: Set[Tuple[Site, Time]] = set()
        self._fifo_list: List[Tuple[Site, Time]] = []

    def fetch_night_events(self, site: Site, time_grid: Time) -> NightEvents:
        """
        Fetch a NightEvents object that contains information for all the nights in
        the time_grid.

        If the cache is too small to hold the time_grid, it will be resized, so theoretically,
        this method should always succeed. If for some reason, it fails on a cache key lookup,
        it will fail with a RuntimeError.
        """
        # If the time_grid is too big that it overwhelms the cache, resize the cache.
        if len(time_grid) > self.cache_size:
            old_size = self.cache_size
            self.cache_size = len(time_grid) * len(Site)
            logging.info(f'Resizing the NightEventsCache from {old_size} to {self.cache_size}.')

        # Make sure that everything is calculated and cached.
        for day in time_grid:
            self._store_night_event(site, day)

        try:
            data = [[cache[(site, time)] for time in time_grid] for cache in self.caches]
            return NightEvents(*data)

        except KeyError as e:
            msg = f'Could not locate the night events for: {e}'
            logging.error(msg)
            raise RuntimeError(msg)

    def _store_night_event(self, site: Site, time: Time) -> NoReturn:
        """
        Calculate the NightEvents for the night specified in the AstroPy Time object.
        This creates the cache entry and maintains the size of the cache.
        """
        if (site, time) not in self._fifo_entries:
            local_timezone = site.value.time_zone
            location = site.value.location

            # See if we need to bump something from the cache.
            if len(self._fifo_list) == self.cache_size:
                # Chop off the first entry.
                key = self._fifo_list.pop(0)
                self._fifo_entries.remove(key)
                for cache in self.caches:
                    del [cache[key]]

            # Calculate the new values and insert them into the caches.
            results = vskyutil.nightevents(time, location, local_timezone, verbose=False)

            key = site, time
            for cache_idx, cache in enumerate(self.caches):
                cache[key] = results[cache_idx]

            # Record
            self._fifo_entries.add(key)
            self._fifo_list.append(key)


@dataclass
class Collector(SchedulerComponent):
    """
    At this point, we still work with AstroPy Time for efficiency.
    We will switch do datetime and timedelta by the end of the Collector
    so that the Scheduler relies on regular Python datetime and timedelta
    objects instead.
    """
    sites: FrozenSet[Site]
    semesters: FrozenSet[Semester]
    program_types: FrozenSet[ProgramTypes]
    obs_classes: FrozenSet[ObservationClass]
    start_time: Time
    end_time: Time
    time_slot_length: TimeDelta

    # This should not be populated, but we put it here instead of in __post_init__
    # to eliminate warnings.
    programs: Dict[Site, Dict[str, Program]] = field(default_factory=lambda: defaultdict(Dict[str, Program]))

    # The NightEventsCache to be used with this collector.
    _NIGHT_EVENTS_CACHE: ClassVar[NightEventsCache] = NightEventsCache()

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
        self.night_events = {site: self._NIGHT_EVENTS_CACHE.fetch_night_events(site, self.time_grid)
                             for site in self.sites}

        # Create the time array, which is an array that represents the minimum starting
        # time to the maximum starting time, divided into segments of length time_slot_length.
        # We want one entry per time slot grid, i.e. per night, measured in UTC, local, and local sidereal.
        self.utc_times = []
        self.local_times = {}
        self.local_sidereal_times = {}

        # TODO: This is across all sites. Do we want this, or should it be site-specific between
        # TODO: sunset and sunrise?
        for i in range(len(self.time_grid)):
            time_min = min([self._MAX_NIGHT_EVENT_TIME] + [self.night_events[site].twi_eve12[i] for site in self.sites])
            time_max = max([self._MIN_NIGHT_EVENT_TIME] + [self.night_events[site].twi_mor12[i] for site in self.sites])
            time_start = helpers.round_minute(time_min, up=True)
            time_end = helpers.round_minute(time_max, up=False)
            time_slot_length_days = self.time_slot_length.to(u.day).value
            n = np.int((time_end.jd - time_start.jd) / time_slot_length_days + 0.5)

            # TODO: Verify that this is all correct.
            time = Time(np.linspace(time_start.jd, time_end.jd - time_slot_length_days, n), format='jd')
            self.utc_times.append(time.to_datetime('utc'))
            for site in Site:
                self.local_times[site].append(time.to_datetime(site.value.timezone))
                self.local_sidereal_times[site].append(vskyutil.lpsidereal(time, site.value.location))

        # Create lists / dicts corresponding to the time grid with an AstroPy SkyCoord array giving
        # the sun or moon position at each time step. Can access via ra and dec members.
        # We also need to convert to alt-az for each site. This all works nicely with AstroPy Time objects.
        # TODO: Local time or UTC?
        # aa_converters = {site: AltAz(location=site.location, obstime=time)
        #                  for site in Site for time in self.local_times}
        # TODO: Using a Time containing an array for obstime in altaz_converters will cause errors if we
        # TODO: use it in conjunction with a SkyCoord containing an array.
        # TODO: We can use EITHER a Time array OR a SkyCoord array, but not both.
        # TODO: We may need to take two approaches to mix them.
        self.altaz_converters = {site: AltAz(location=site.location, obstime=self.local_times) for site in Site}

        # Sun positions in RA / Dec and then for each site in Alt / Az.
        self.sun_position_radec = vskyutil.lpsun(self.utc_times)
        self.sun_position_altaz = {site: self.sun_position_radec.transform_to(self.altaz_converters[site])
                                   for site in Site}

        # Lunar position in RA / Dec and then for each site in Alt / Az.
        # This will flag a warning because lpmoon requests a float but Bryan has vectorized it.
        self.moon_position_radec = {site: vskyutil.lpmoon(self.local_times, site.value.location) for site in Site}
        self.moon_position_altaz = {site: self.moon_position_radec[site].transform_to(self.altaz_converters[site])
                                    for site in Site}

        # Lunar distance and altitude.
        # This is a tuple (SkyCoord, distance) for the site at the given time array.
        # This will flag a warning because lpmoon requests a float but Bryan has vectorized it.
        # self.moon_location_and_distance = {site: vskyutil.accumoon(self.local_times, site.value.location)
        #                                    for site in Site}

        # We begin with zero observations.
        self.num_observations = 0

    def load_programs(self, program_provider: ProgramProvider, data: Mapping[Site, Iterable[dict]]) -> NoReturn:
        """
        Load the programs provided as JSON into the Collector.

        The program_provider should be a concrete implementation of the API to read in
        programs from JSON files.

        The json_data comprises the program inputs as an iterable object per site. We use iterable
        since the amount of data here might be enormous, and we do not want to store it all
        in memory at once.
        """
        # Purge the old programs.
        self.programs = {}

        # As we read in the programs, collect the targets for the site.
        # TODO: For nonsidereal targets (not currently handled), we need ephemeris data.
        # For sidereal targets, we have to interpolate over the time period under consideration.
        sidereal_targets: dict[Site, Set[SiderealTarget]] = {}

        for site in data.keys():
            if site not in self.sites:
                # Count the iterable, which consumes it.
                length = sum(1 for _ in data[site])
                logging.warning(f'JSON data contained ignored site {site.name}: {length} programs dropped.')
                continue

            # Read in the programs for the site.
            # We do this using a loop instead of a for comprehension because we want the IDs.
            self.programs[site] = {}

            for json_program in data[site]:
                # Count the number of parse failures.
                bad_program_count = 0

                try:
                    if len(json_program.keys()) > 1:
                        msg = f'JSON programs should only have one top-level key: {" ".join(json_program.keys())}'
                        logging.error(msg)
                        raise ValueError(msg)

                    # Extract the data from the JSON program.
                    label, data = list(json_program.items())[0]

                    program = program_provider.parse_program(data)
                    if program.id in self.programs[site].keys():
                        logging.warning(f'Site {site.name} contains a repeated program with id {program.id}.')
                    self.programs[site][program.id] = program

                    def collect_targets(node_group: NodeGroup) -> NoReturn:
                        """
                        Pull up the information about all targets scheduled at each site.
                        """
                        if isinstance(node_group.children, SiderealTarget):
                            sidereal_targets.setdefault(site, set())
                            sidereal_targets[site].add(node_group.children)
                        elif isinstance(node_group.children, NonsiderealTarget):
                            # TODO: Fill this in.
                            ...
                        else:
                            for subgroup in node_group.children:
                                collect_targets(subgroup)

                    collect_targets(program.root_group)

                except ValueError as e:
                    bad_program_count += 1
                    logging.warning(f'Could not parse program: {e}')

                if bad_program_count:
                    logging.error(f'For site {site.name}, could not parse {bad_program_count} programs.')

        # Now we process the targets per site to get the necessary information.
        # Indexed by site and target name, result is an array of nights with each
        # entry an array of coordinates, either RA or Dec.
        sidereal_target_ra: dict[(Site, str), npt.NDArray[npt.NDArray[float]]] = {}
        sidereal_target_dec: dict[(Site, str), npt.NDArray[npt.NDArray[float]]] = {}

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
                    sidereal_target_ra = target.ra + pm_ra * time_offsets
                    sidereal_target_dec = target.dec + pm_dec * time_offsets
                    sidereal_target_coords = SkyCoord(sidereal_target_ra * u.deg, sidereal_target_dec * u.deg)

                    # Convert to Alt-Az.
                    sidereal_target_altaz = sidereal_target_coords.transform_to(self.altaz_converters[site])

                    # Calculate the hour angles.
                    sidereal_target_hrangle = vskyutil.ha_alt(sidereal_target_dec,
                                                              sidereal_target_coords.lat,
                                                              sidereal_target_altaz.alt)

                    # Calculate the airmasses.
                    sidereal_target_airmass = vskyutil.true_airmass(sidereal_target_altaz.alt)

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
