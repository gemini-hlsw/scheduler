import logging

from astropy.coordinates import Angle
from astropy.time import Time, TimeDelta
import astropy.units as u
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt
from typing import ClassVar, FrozenSet, Iterable, List, Mapping, NoReturn, Set, Tuple

from common.minimodel import ObservationClass, Program, ProgramTypes, Semester, Site
from common.api import ProgramProvider
from common.scheduler import SchedulerComponent
import common.vskyutil as vskyutil


@dataclass
class Collector(SchedulerComponent):
    """
    At this point, we still work with AstroPy Time for efficiency.
    We will switch do datetime and timedelta by the end of the Collector
    so that the Scheduler relies on regular Python datetime and timedelta
    objects instead.

    The program_provider should be a concrete implementation of the API to read in
    programs from JSON files.

    The json_data comprises the program inputs as an iterable object per site. We use iterable
    since the amount of data here might be enormous and we do not want to store it all
    in memory at once.
    """
    sites: FrozenSet[Site]
    semesters: FrozenSet[Semester]
    program_types: FrozenSet[ProgramTypes]
    obs_classes: FrozenSet[ObservationClass]
    start_time: Time
    end_time: Time
    time_slot_length: TimeDelta
    program_provider: ProgramProvider

    # This is too generic to be typed as anything else.
    json_data: Mapping[Site, Iterable[dict]]

    # The default timeslot length currently used.
    DEFAULT_TIMESLOT_LENGTH: ClassVar[Time] = 1.0 * u.min

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

        # Load the programs. Ignore any programs that are not in the list of sites.
        self.programs: dict[Site, dict[str, Program]] = {}
        for site in self.json_data.keys():
            if site not in self.sites:
                # Count the iterable, which consumes it.
                length = sum(1 for _ in self.json_data[site])
                logging.warning(f'JSON data contained ignored site {site.name}: {length} programs dropped.')
                continue

            # Read in the programs for the site.
            # We do this using a loop instead of a for comprehension because we want the IDs.
            self.programs[site] = {}
            for json_program in self.json_data[site]:
                # Count the number of parse failures.
                bad_program_count = 0

                try:
                    program = self.program_provider.parse_program(json_program)
                    if program.id in self.programs[site].keys():
                        logging.warning(f'Site {site.name} contains a repeated program with id {program.id}.')
                    self.programs[site][program.id] = program

                # TODO: Can we make this less generic? Specify which exceptions to include?
                except:
                    bad_program_count += 1

                if bad_program_count:
                    logging.error(f'For site {site.name}, could not parse {bad_program_count} programs.')

        # Create the night events, which contain the data for all given nights by site.
        self.night_events = {site: self._calculate_night_events(site) for site in self.sites}


    # TODO: These are exclusive to the create_time_array, which is found in the Selector.
    # TODO: They should probably be moved there.
    _MIN_NIGHT_EVENT_TIME = ClassVar[Time('1980-01-01 00:00:00', format='iso', scale='utc')]
    _MAX_NIGHT_EVENT_TIME = ClassVar[Time('2200-01-01 00:00:00', format='iso', scale='utc')]

    def create_time_array(self):
        """
        This is used by the Selector and should probably be moved there.
        It is only called if the times parameter to the Selector is None.
        TODO: Do we need to use astropy units at this point? I don't see how not.
        """
        times_array = []
        slot_length_value = self.time_slot_length.to(u.day).value

        for idx in range(len(self.time_grid)):
            twi_min = min([Collector._MAX_NIGHT_EVENT_TIME] + [self.night_events[site].twi_eve12[idx]
                                                               for site in self.sites])
            twi_max = min([Collector._MIN_NIGHT_EVENT_TIME] + [self.night_events[site].twi_mor12[idx]
                                                               for site in self.sites])
            twi_start = Collector._round_minute(twi_min, up=True)
            twi_end = Collector._round_minute(twi_max, up=False)
            num = np.int((twi_end.jd - twi_start.jd) / slot_length_value + 0.5)
            times_array.append(Time(np.linspace(twi_start.jd, twi_end.jd - slot_length_value, num), format='jd'))

        return times_array

    def _calculate_night_events(self, site: Site):
        """
        Attempt to load a cached instance of the night events for the time grid
        and the given site. If they do not exist, create a new cache entry.
        """
        return NightEvents.get_night_events(site, self.time_grid)

    @staticmethod
    def _round_minute(time: Time, up: bool) -> Time:
        """
        Round a time down (truncate) or up to the nearest minute

        time: an astropy.Time
        up: bool indicating whether to round up
        """
        t = time.copy()
        t.format = 'iso'
        t.out_subfmt = 'date_hm'
        if up:
            sec = int(t.strftime('%S'))
            if sec:
                t += 1.0 * u.min
        return Time(t.iso, format='iso', scale='utc')


@dataclass
class NightEvents:
    """
    Represents night events for the period under consideration as represented
    by the time. NightEvents should be managed by NightEventsCache for efficiency.
    """
    midnight: Time
    sunset: Time
    sunrise: Time
    twi_eve18: Time
    twi_mor18: Time
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
class NightEventCache:
    """
    A cache of NightEvents for a given night.
    We cache as these are expensive computations and we wish to avoid repeated
    calculations.
    """
    # By default, just allow the equivalent of a year's worth of data for each site.
    cache_size = 365 * len(Site)

    def __post_init__(self):
        """
        Initialize the caches, which are FIFO (for now).
        """
        self._midnight_cache: dict[Tuple[Site, Time], Time] = {}
        self._sunset_cache: dict[Tuple[Site, Time], Time] = {}
        self._sunrise_cache: dict[Tuple[Site, Time], Time] = {}
        self._twi_eve18_cache: dict[Tuple[Site, Time], Time] = {}
        self._twi_mor18_cache: dict[Tuple[Site, Time], Time] = {}
        self._twi_eve12_cache: dict[Tuple[Site, Time], Time] = {}
        self._twi_mor12_cache: dict[Tuple[Site, Time], Time] = {}
        self._moonrise_cache: dict[Tuple[Site, Time], Time] = {}
        self._moonset_cache: dict[Tuple[Site, Time], Time] = {}
        self._sun_moon_ang_cache: dict[Tuple[Site, Time], Angle] = {}
        self._moon_illum_cache: dict[Tuple[Site, Time], npt.NDArray[float]] = {}
        self.caches = {}

        self._fifo_entries: Set[Tuple[Site, Time]] = set()
        self._fifo_list: List[Tuple[Site, Time]] = []

    def fetch_night_events(self, site: Site, time_grid: Time) -> NightEvents:
        """
        Fetch a NightEvents object that contains information for all of the nights in
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
        for time in time_grid:
            self._store_night_event(site, time)

        try:
            midnight = Time(self._midnight_cache[(site, day)] for day in time_grid)
            sunset = Time(self._sunset_cache[(site, day)] for day in time_grid)
            sunrise = Time(self._sunrise_cache[(site, day)] for day in time_grid)
            twi_eve18 = Time(self._twi_eve18_cache[(site, day)] for day in time_grid)
            twi_mor18 = Time(self._twi_mor18_cache[(site, day)] for day in time_grid)
            twi_eve12 = Time(self._twi_eve12_cache[(site, day)] for day in time_grid)
            twi_mor12 = Time(self._twi_mor12_cache[(site, day)] for day in time_grid)
            moonrise = Time(self._moonrise_cache[(site, day)] for day in time_grid)
            moonset = Time(self._moonset_cache[(site, day)] for day in time_grid)
            sun_moon_ang = Angle(self._sun_moon_ang_cache[(site, day)] for day in time_grid)
            moon_illum = np.ndarray(self._moon_illum_cache[(site, day)] for day in time_grid)
            return NightEvents(midnight, sunset, sunrise, twi_eve18, twi_mor18, twi_eve12, twi_mor12, moonrise, moonset,
                               sun_moon_ang, moon_illum)

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
                data = self._fifo_list.pop(0)
                self._fifo_entries.remove(data)
                del[self._midnight_cache[data]]
                del[self._sunset_cache[data]]
                del[self._sunrise_cache[data]]
                del[self._twi_eve18_cache[data]]
                del[self._twi_mor18_cache[data]]
                del[self._twi_eve12_cache[data]]
                del[self._twi_mor12_cache[data]]
                del[self._moonrise_cache[data]]
                del[self._sun_moon_ang_cache[data]]
                del[self._moon_illum_cache[data]]

            # Calculate the new values and insert them into the caches.
            midnight, sunset, sunrise, twi_eve18, twi_mor18, twi_eve12, twi_mor12,\
                moonrise, sun_moon_ang, moon_illum = vskyutil.nightevents(time, location, local_timezone, verbose=False)

            key = site, time
            self._midnight_cache[key] = midnight
            self._sunset_cache[key] = sunset
            self._sunrise_cache[key] = sunrise
            self._twi_eve18_cache[key] = twi_eve18
            self._twi_mor18_cache[key] = twi_mor18
            self._twi_eve12_cache[key] = twi_eve12
            self._twi_mor12_cache[key] = twi_mor12
            self._moonrise_cache[key] = moonrise
            self._moon_illum_cache[key] = moon_illum

            # Record
            self._fifo_entries.add(key)
            self._fifo_list.append(key)
