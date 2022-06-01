import logging
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
import pytz

from common.minimodel import *
from common import helpers
from common import sky

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

# Scores for the timeslots in a specific night.
NightTimeSlotScores = npt.NDArray[float]

# Scores across all nights for the timeslots.
Scores = List[NightTimeSlotScores]


@dataclass(frozen=True)
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
        night_length = (self.twilight_morning_12 - self.twilight_evening_12).to_value('h') * u.h
        object.__setattr__(self, 'night_length', night_length)

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
        times = [Time(np.linspace(start.jd, end.jd - time_slot_length_days, i), format='jd')
                 for start, end, i in zip(time_starts, time_ends, n)]
        object.__setattr__(self, 'times', times)

        # Pre-calculate the different times.
        # We want these as Python lists because the entries will have different lengths.
        utc_times = [t.to_datetime(pytz.UTC) for t in times]
        object.__setattr__(self, 'utc_times', utc_times)

        local_times = [t.to_datetime(self.site.value.timezone) for t in times]
        object.__setattr__(self, 'local_times', local_times)

        # self.local_sidereal_times = [vskyutil.lpsidereal(t, self.site.value.location) for t in self.times]
        local_sidereal_times = [sky.local_sidereal_time(t, self.site.value.location) for t in times]
        object.__setattr__(self, 'local_sidereal_times', local_sidereal_times)

        def altazparang(pos: List[SkyCoord]) -> Tuple[npt.NDArray[Angle], npt.NDArray[Angle], npt.NDArray[Angle]]:
            """
            Common code to invoke vskyutil.altazparang for a number of positions and then properly
            combine the results into three numpy arrays, representing, indexed by time:
            1. altitude
            2. azimuth
            3. parallactic angle
            """
            alt, az, par_ang = zip(
                *[sky.Altitude.above(p.dec, lst - p.ra, self.site.value.location.lat)
                  for p, lst in zip(pos, local_sidereal_times)]
            )

            return alt, az, par_ang

        # Calculate the parameters for the sun, joining together the positions.
        # We also precalculate the indices for the time slots for the night that have the required sun altitude.
        sun_pos = [SkyCoord(sky.Sun.at(t)) for t in times]
        object.__setattr__(self, 'sun_pos', sun_pos)

        sun_alt, sun_az, sun_par_ang = altazparang(sun_pos)
        object.__setattr__(self, 'sun_alt', sun_alt)
        object.__setattr__(self, 'sun_az', sun_az)
        object.__setattr__(self, 'sun_par_ang', sun_par_ang)

        sun_alt_indices = [np.where(sun_alt[night_idx] <= -12 * u.deg)[0] for night_idx, _ in enumerate(self.time_grid)]
        object.__setattr__(self, 'sun_alt_indices', sun_alt_indices)

        # accumoon produces a tuple, (SkyCoord, ndarray) indicating position and distance.
        # In order to populate both moon_pos and moon_dist, we use the zip(*...) technique to
        # collect the SkyCoords into one tuple, and the ndarrays into another.
        # The moon_dist are already a Quantity: error if try to convert.
        moon_pos, moon_dist = zip(*[sky.Moon().at(t).accurate_location(self.site.value.location) for t in times])
        object.__setattr__(self, 'moon_pos', moon_pos)
        object.__setattr__(self, 'moon_dist', moon_dist)

        moon_alt, moon_az, moon_par_ang = altazparang(moon_pos)
        object.__setattr__(self, 'moon_alt', moon_alt)
        object.__setattr__(self, 'moon_az', moon_az)
        object.__setattr__(self, 'moon_par_ang', moon_par_ang)

        # Angle between the sun and the moon.
        sun_moon_ang = [sun_pos.separation(moon_pos) for sun_pos, moon_pos in zip(sun_pos, moon_pos)]
        object.__setattr__(self, 'sun_moon_ang', sun_moon_ang)

        # List of numpy arrays used to calculate proper motion of sidereal targets for the night.
        pm_array = [(t.value - NightEvents._JULIAN_BASIS) / NightEvents._JULIAN_YEAR_LENGTH for t in times]
        object.__setattr__(self, 'pm_array', pm_array)


@dataclass(frozen=True)
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
TargetInfoNightIndexMap = Dict[NightIndex, TargetInfo]
TargetInfoMap = Dict[Tuple[TargetName, ObservationID], TargetInfoNightIndexMap]


@dataclass(frozen=True)
class GroupInfo:
    """
    Information regarding Groups that can only be calculated in the Selector.

    Note that the lists here are indexed by night indices as passed to the selection method, or
      equivalently, as defined in the Ranker.

    This comprises:
    1. The most restrictive Conditions required for the group as per all its subgroups.
    2. The slots in which the group can be scheduled based on resources and environmental conditions.
    3. The score assigned to the group.
    4. The standards time associated with the group, in hours.
    5. A flag to indicate if the group can be split.
    A group can be split if and only if it contains more than one observation.
    """
    minimum_conditions: Conditions
    is_splittable: bool
    standards: float
    resource_night_availability: npt.NDArray[bool]
    conditions_score: List[npt.NDArray[float]]
    wind_score: List[npt.NDArray[float]]
    schedulable_slot_indices: List[npt.NDArray[int]]
    scores: Scores


@dataclass(frozen=True)
class GroupData:
    group: Group
    group_info: GroupInfo


# This has to be a Dict because we need to be able to write to it using [idx].
GroupDataMap = Dict[GroupID, GroupData]


@dataclass(frozen=True)
class ProgramInfo:
    """
    This represents the information for a program that contains schedulable components during the time frame
    under consideration, along with those schedulable components.
    """
    program: Program

    # Schedulable groups by ID and their information.
    group_data: GroupDataMap

    # Schedulable observations by their ID. This is duplicated in the group information above
    # but provided for convenience.
    observations: Mapping[ObservationID, Observation]

    # Target information relevant to this program.
    target_info: Mapping[ObservationID, TargetInfoNightIndexMap]

    def __post_init__(self):
        # Set up the keys from the data. We have to use this ugly syntax to make the dataclass frozen.
        object.__setattr__(self, 'observation_ids', frozenset(self.observations.keys()))
        object.__setattr__(self, 'group_ids', frozenset(self.group_data.keys()))


@dataclass(frozen=True)
class Selection:
    """
    The selection of information passed by the Selector to the Optimizer.
    This includes the list of programs that are schedulable
    """
    program_info: Mapping[ProgramID, ProgramInfo]
    night_events: Mapping[Site, NightEvents]

    def __post_init__(self):
        # Set up the keys from the data.
        object.__setattr__(self, 'program_ids', frozenset(self.program_info.keys()))
