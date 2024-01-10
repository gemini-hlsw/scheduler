# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytz
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time, TimeDelta
from lucupy import helpers, sky
from lucupy.decorators import immutable
from lucupy.minimodel import NightIndex, Site, TimeslotIndex
from lucupy.sky.constants import JYEAR, J2000


@immutable
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

    # post-init calculated values.
    night_length: TimeDelta = field(init=False)
    times: List[npt.NDArray[float]] = field(init=False)
    utc_times: List[List[datetime]] = field(init=False)
    local_times: List[List[datetime]] = field(init=False)
    local_sidereal_times: List[Angle] = field(init=False)

    sun_pos: List[SkyCoord] = field(init=False)
    sun_alt: List[npt.NDArray[Angle]] = field(init=False)
    sun_az: List[npt.NDArray[Angle]] = field(init=False)
    sun_par_ang: List[npt.NDArray[Angle]] = field(init=False)
    sun_alt_indices: List[npt.NDArray[int]] = field(init=False)

    moon_pos: List[SkyCoord] = field(init=False)
    moon_dist: List[float] = field(init=False)
    moon_alt: List[npt.NDArray[Angle]] = field(init=False)
    moon_az: List[npt.NDArray[Angle]] = field(init=False)
    moon_par_ang: List[npt.NDArray[Angle]] = field(init=False)

    sun_moon_ang: List[Angle] = field(init=False)

    pm_array: List[npt.NDArray[float]] = field(init=False)

    def __post_init__(self):
        # Calculate the length of each night at this site, i.e. time between twilights.
        night_length = TimeDelta((self.twilight_morning_12 - self.twilight_evening_12).to(u.hour))
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

        local_times = [t.to_datetime(self.site.timezone) for t in times]
        object.__setattr__(self, 'local_times', local_times)

        local_sidereal_times = [sky.local_sidereal_time(t, self.site.location) for t in times]
        object.__setattr__(self, 'local_sidereal_times', local_sidereal_times)

        def alt_az_parang(pos: List[SkyCoord]) -> Tuple[npt.NDArray[Angle], npt.NDArray[Angle], npt.NDArray[Angle]]:
            """
            Common code to invoke vskyutil.altazparang for a number of positions and then properly
            combine the results into three numpy arrays, representing, indexed by time:
            1. altitude
            2. azimuth
            3. parallactic angle
            """
            alt, az, par_ang = zip(
                *[sky.Altitude.above(p.dec, lst - p.ra, self.site.location.lat)
                  for p, lst in zip(pos, local_sidereal_times)]
            )

            return alt, az, par_ang

        # Calculate the parameters for the sun, joining together the positions.
        # We also precalculate the indices for the time slots for the night that have the required sun altitude.
        sun_pos = [SkyCoord(sky.Sun.at(t)) for t in times]
        object.__setattr__(self, 'sun_pos', sun_pos)

        sun_alt, sun_az, sun_par_ang = alt_az_parang(sun_pos)
        object.__setattr__(self, 'sun_alt', sun_alt)
        object.__setattr__(self, 'sun_az', sun_az)
        object.__setattr__(self, 'sun_par_ang', sun_par_ang)

        sun_alt_indices = [np.where(sun_alt[night_idx] <= -12 * u.deg)[0] for night_idx, _ in enumerate(self.time_grid)]
        object.__setattr__(self, 'sun_alt_indices', sun_alt_indices)

        # Produces a tuple, (SkyCoord, float) indicating position and distance.
        # TODO: No idea what the unit of distance is.
        # In order to populate both moon_pos and moon_dist, we use the zip(*...) technique to
        # collect the SkyCoords into one tuple, and the floats into another.
        moon_pos, moon_dist = zip(*[sky.Moon().at(t).accurate_location(self.site.location) for t in times])
        object.__setattr__(self, 'moon_pos', moon_pos)
        object.__setattr__(self, 'moon_dist', moon_dist)

        moon_alt, moon_az, moon_par_ang = alt_az_parang(moon_pos)
        object.__setattr__(self, 'moon_alt', moon_alt)
        object.__setattr__(self, 'moon_az', moon_az)
        object.__setattr__(self, 'moon_par_ang', moon_par_ang)

        # Angle between the sun and the moon.
        sun_moon_ang = [sun_pos.separation(moon_pos) for sun_pos, moon_pos in zip(sun_pos, moon_pos)]
        object.__setattr__(self, 'sun_moon_ang', sun_moon_ang)

        # List of numpy arrays used to calculate proper motion of sidereal targets for the night.
        pm_array = [(t.value - J2000) / JYEAR for t in times]
        object.__setattr__(self, 'pm_array', pm_array)

    def utc_dt_to_time_coords(self, dt: datetime) -> Optional[Tuple[NightIndex, TimeslotIndex]]:
        """
        Given a datetime, set the timezone to UTC and determine its time coordinates as a NightIndex and TimeslotIndex
        if possible.
        If it does not correspond to a NightIndex or TimeslotIndex, return None.
        """
        dt = dt.replace(tzinfo=pytz.UTC)
        return dt_to_time_coords(dt, self.time_slot_length.to_datetime(), self.utc_times)

    def local_dt_to_time_coords(self, dt: datetime) -> Optional[Tuple[NightIndex, TimeslotIndex]]:
        """
        Given a (local) datetime, convert it to a NightIndex and a TimeslotIndex if possible.
        If the timezone associated with the datetime is None, the timezone is set.
        If it is anything other than the timezone associated with the site, a ValueError is raised.
        If there is no corresponding NightIndex or TimeslotIndex, None is returned.
        """
        if dt.tzinfo is None:
            # Even if this is the wonky LMT from pytz, this seems to localize properly.
            dt = self.site.timezone.localize(dt)

        # To work around the LMT value from pytz, we localize now to get the proper timezone so that this call does
        # not fail.
        expected_tz = self.site.timezone.localize(datetime.now()).tzinfo
        if dt.tzinfo != expected_tz:
            raise ValueError(f'Expected timezone {expected_tz}, got {dt.tzinfo}.')
        return dt_to_time_coords(dt, self.time_slot_length.to_datetime(), self.local_times)

    def time_coords_to_local_dt(self, night_idx: NightIndex, timeslot_idx: TimeslotIndex) -> Optional[datetime]:
        """
        Given time coordinates, convert to a local datetime.
        """
        return time_coords_to_dt(night_idx, timeslot_idx, self.local_times)

    def time_coords_to_utc_dt(self, night_idx: NightIndex, timeslot_idx: TimeslotIndex) -> Optional[datetime]:
        """
        Given time coordinates, convert to UTC datetime.
        """
        return time_coords_to_dt(night_idx, timeslot_idx, self.utc_times)


def dt_to_time_coords(dt: datetime,
                      timeslot_length: timedelta,
                      times: List[List[datetime]]) -> Optional[Tuple[NightIndex, TimeslotIndex]]:
    """
    Given:
    * a datetime
    * an array of arrays of datetime where the flattened array would be sorted (representing night indices and
        times of timeslots during the night)
    If dt falls within the ranges represented by times, return the NightIndex and TimeslotIndex which corresponds.
    If no such value is found, return None.
    """

    # Given that we rounded all times up, if dt is:
    # * before the first eve twi by at least time_slot_length; or
    # * after the end of the last twilight;
    # then there is no corresponding night index.
    if dt < times[0][0] - timeslot_length or dt >= times[-1][-1]:
        return None

    # Binary search to find the night_idx in which dt falls, if any.
    # Since we rounded times up for each night, we use bisect_left on the end times per night.
    night_idx = bisect.bisect_left([night[-1] for night in times], dt)
    night_times = times[night_idx]

    # Binary search to find the timeslot_idx in which dt falls.
    timeslot_idx = bisect.bisect_left(night_times, dt)
    if timeslot_idx == len(night_times) or dt <= night_times[timeslot_idx] - timeslot_length:
        return None

    return NightIndex(night_idx), TimeslotIndex(timeslot_idx)


def time_coords_to_dt(night_idx: NightIndex,
                      timeslot_idx: TimeslotIndex,
                      times: List[List[datetime]]) -> Optional[datetime]:
    """
    Given night coordinates (a night index and a timeslot index) and an array of night data, convert the time
    coordinates to a time in the night data.
    If no such time exists, return None.
    """
    if 0 <= night_idx < len(times):
        night_times = times[night_idx]
        if 0 <= timeslot_idx < len(night_times):
            return night_times[timeslot_idx]
    return None
