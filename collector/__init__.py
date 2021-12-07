import logging
from astropy.time import Time, TimeDelta
import astropy.units as u
from dataclasses import dataclass
import numpy as np
from typing import ClassVar, FrozenSet, Mapping, Set
from common.minimodel import NightEvents, ObservationClass, Program, ProgramTypes, Semester, Site
from common.api import ProgramProvider
from common.scheduler import SchedulerComponent


@dataclass
class Collector(SchedulerComponent):
    """
    At this point, we still work with AstroPy Time for efficiency.
    We will switch do datetime and timedelta by the end of the Collector
    so that the Scheduler relies on regular Python datetime and timedelta
    objects instead.

    TODO: Most of this class is read in from XML in the current implementation and values are
    TODO: hardcoded for the example.
    TODO: Instead, we will need to read it in from JSON files, which will require considerable
    TODO: redesign.
    """
    sites: FrozenSet[Site]
    semesters: FrozenSet[Semester]
    program_types: FrozenSet[ProgramTypes]
    obs_classes: FrozenSet[ObservationClass]
    start_time: Time
    end_time: Time
    time_slot_length: TimeDelta
    program_provider: ProgramProvider
    json: dict

    # Counter to keep track of observations.
    _num_observations: ClassVar[int] = 0

    def __post_init__(self):
        """
        Initializes the internal data structures for the Collector and populates them.
        """
        # Set up the time grid for the period under consideration: this is an astropy Time
        # object from start_time to end_time inclusive, spaced apart by time_slot_length.
        self.time_grid = self._create_time_grid()

        # Create an empty placeholder for programs by site.
        # TODO: Do we really need to index by program id or can we just use a Set of program?
        self.programs: Mapping[Site, Mapping[str, Program]] = {site: set() for site in self.sites}

        # Create the night events, which contain the data for all given nights by site.
        # TODO: This should be precomputed and cached for start_time to end_time, I think, for performance.
        self.night_events = {site: self._calculate_night_events(site) for site in self.sites}

    def _create_time_grid(self) -> Time:
        """
        Creates the time grid for the Collector to use.

        This initializes a numpy array of datetime objects from start_time to end_time inclusive,
        each of size time_slot_length.

        ValueError is raised if:
        * the start_time does not occur before the end_time
        * the time_slot_length is not positive
        """
        # The start time must be before the end time.
        if self.start_time >= self.end_time:
            msg = f'Start time ({self.start_time}) must be earlier than end time ({self.end_time}).'
            logging.error(msg)
            raise ValueError(msg)

        # The time_slot_length must be positive.
        if self.time_slot_length <= TimeDelta():
            msg = f'Time slot length {self.time_slot_length} must be positive.'
            logging.error(msg)
            raise ValueError(msg)

        # Create the time grid.
        return Time(np.arange(self.start_time.jd, self.end_time.jd + 1.0, (1.0 * u.day).value), format='jd')

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

    @staticmethod
    def num_observations() -> int:
        """
        Return the number of observations in the collector.
        TODO: Do we care about the number of observations at this point, or the atoms?
        """
        return Collector._num_observations

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
