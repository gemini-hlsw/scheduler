import logging

from astropy.coordinates import Angle
from astropy.time import Time, TimeDelta
from astropy import units as u
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
import numpy.typing as npt
from typing import ClassVar, Dict, FrozenSet, Iterable, List, Mapping, NoReturn, Set, Tuple

from common.api import ProgramProvider
import common.helpers as helpers
from common.minimodel import ObservationClass, Program, ProgramTypes, Semester, Site
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
        for day in time_grid:
            self._store_night_event(site, day)

        try:
            data = (cache[(site, day)] for day in time_grid for cache in self.caches)
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
                    del[cache[key]]

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
        # We want one entry per time slot grid, i.e. per night.
        self.times = []
        for i in range(len(self.time_grid)):
            time_min = min([self._MAX_NIGHT_EVENT_TIME] + [self.night_events[site].twi_eve12[i] for site in self.sites])
            time_max = max([self._MIN_NIGHT_EVENT_TIME] + [self.night_events[site].twi_mor12[i] for site in self.sites])
            time_start = helpers.round_minute(time_min, up=True)
            time_end = helpers.round_minute(time_max, up=False)

            time_slot_length_days = self.time_slot_length.to(u.day).value
            n = np.int((time_end.jd - time_start.jd) / time_slot_length_days * 0.5)
            self.times.append(Time(np.linspace(time_start.jd, time_end.jd - time_slot_length_days, n), format='jd'))

        # We begin with zero observations.
        self.num_observations = 0

    def load_programs(self, program_provider: ProgramProvider, json_data: Mapping[Site, Iterable[dict]]) -> NoReturn:
        """
        Load the programs provided as JSON into the Collector.

        The program_provider should be a concrete implementation of the API to read in
        programs from JSON files.

        The json_data comprises the program inputs as an iterable object per site. We use iterable
        since the amount of data here might be enormous and we do not want to store it all
        in memory at once.
        """
        # Purge the old programs.
        self.programs = {}

        for site in json_data.keys():
            if site not in self.sites:
                # Count the iterable, which consumes it.
                length = sum(1 for _ in json_data[site])
                logging.warning(f'JSON data contained ignored site {site.name}: {length} programs dropped.')
                continue

            # Read in the programs for the site.
            # We do this using a loop instead of a for comprehension because we want the IDs.
            self.programs[site] = {}
            for json_program in json_data[site]:
                # Count the number of parse failures.
                bad_program_count = 0

                try:
                    program = program_provider.parse_program(json_program)
                    if program.id in self.programs[site].keys():
                        logging.warning(f'Site {site.name} contains a repeated program with id {program.id}.')
                    self.programs[site][program.id] = program

                    # Now extract the observation and target information from the program.
                    # TODO

                except ValueError as e:
                    bad_program_count += 1
                    logging.warning(f'Could not parse program: {e}')

                if bad_program_count:
                    logging.error(f'For site {site.name}, could not parse {bad_program_count} programs.')
