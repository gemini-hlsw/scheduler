from datetime import date, datetime, timezone, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo
import numpy as np
import numpy.typing as npt
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from pydantic import BaseModel

import lucupy.sky as sky

from scheduler.services.sight.calculations.arrays import pack_array

if TYPE_CHECKING:
    from scheduler.services.sight.database.models import Site


class NightEventArrays(BaseModel):
    """
    Night event data ready for database storage.
    
    All arrays are packed as bytes.
    All angles are in radians.
    """
    # Scalars
    night_duration_minutes: int
    sunset: datetime
    sunrise: datetime
    night_start: datetime
    night_end: datetime
    midnight: datetime
    twilight_evening_12: datetime
    twilight_morning_12: datetime
    moonrise: datetime | None
    moonset: datetime | None
    moon_dist: float  # AU
    
    # Packed arrays (bytes)
    sun_alt: bytes
    sun_az: bytes
    sun_par_ang: bytes
    moon_alt: bytes
    moon_az: bytes
    moon_par_ang: bytes
    moon_ra: bytes   # degrees
    moon_dec: bytes  # degrees
    sun_moon_ang: bytes
    local_sidereal_times: bytes
    
    class Config:
        arbitrary_types_allowed = True


def site_to_earth_location(site: "Site") -> EarthLocation:
    """Convert Site model to astropy EarthLocation."""
    return EarthLocation(
        lat=site.latitude * u.deg,
        lon=site.longitude * u.deg,
        height=site.elevation * u.m,
    )


def astropy_time_to_datetime(t: Time) -> datetime:
    """Convert astropy Time to UTC datetime."""
    return t.to_datetime(timezone=timezone.utc)


def calculate_night_events_for_night(
    site: "Site",
    night_date: date,
    time_slot_length_minutes: int = 1,
) -> NightEventArrays:
    """
    Calculate night events for a single night at a site.
    
    Args:
        site: Site model with lat/lon/elevation
        night_date: Date of the night (evening date)
        time_slot_length_minutes: Time slot granularity
        
    Returns:
        NightEventArrays ready for database storage
    """
    location = site_to_earth_location(site)
    
    # Create time grid for single night
    time_grid = Time([datetime(night_date.year, night_date.month, night_date.day)])
    
    # Get timestamps from sky.night_events (returns tuple)
    midnight, sunset, sunrise, twilight_evening_12, twilight_morning_12, moonrise, moonset = \
        sky.night_events(time_grid, location, ZoneInfo('UTC'))
    
    # Extract single night values (index 0)
    midnight_0 = midnight[0]
    sunset_0 = sunset[0]
    sunrise_0 = sunrise[0]
    twi_eve_0 = twilight_evening_12[0]
    twi_mor_0 = twilight_morning_12[0]
    moonrise_0 = moonrise[0] if moonrise is not None else None
    moonset_0 = moonset[0] if moonset is not None else None
    
    # Calculate time slots
    time_slot_length = TimeDelta(time_slot_length_minutes * 60, format='sec')
    timeslot_length_days = time_slot_length.to(u.day).value
    
    # Round to nearest minute
    time_start = _round_minute(twi_eve_0, up=True)
    time_end = _round_minute(twi_mor_0, up=True)
    
    # Number of time slots
    num_slots = int((time_end.jd - time_start.jd) / timeslot_length_days + 0.5)
    
    # Create time array for this night
    times = Time(np.linspace(time_start.jd, time_end.jd - timeslot_length_days, num_slots), format='jd')
    
    # Calculate local sidereal times
    local_sidereal_times = sky.local_sidereal_time(times, location)
    
    # Calculate sun position and alt/az/par_ang
    sun_pos = SkyCoord(sky.Sun.at(times))
    sun_alt, sun_az, sun_par_ang = sky.Altitude.above(
        sun_pos.dec, 
        local_sidereal_times - sun_pos.ra, 
        location.lat
    )
    
    # Calculate moon position and alt/az/par_ang
    moon_pos, moon_dist_obj = sky.Moon().at(times).accurate_location(location)
    moon_alt, moon_az, moon_par_ang = sky.Altitude.above(
        moon_pos.dec,
        local_sidereal_times - moon_pos.ra,
        location.lat
    )
    
    # Extract moon distance in AU
    moon_dist_au = moon_dist_obj.to(u.AU).value
    if hasattr(moon_dist_au, '__len__'):
        moon_dist_au = float(np.mean(moon_dist_au))
    else:
        moon_dist_au = float(moon_dist_au)
    
    # Sun-moon angular separation
    sun_moon_ang = sun_pos.separation(moon_pos)
    
    # Convert to numpy arrays
    sun_alt_rad = _angle_to_radians(sun_alt)
    sun_az_rad = _angle_to_radians(sun_az)
    sun_par_ang_rad = _angle_to_radians(sun_par_ang)
    
    moon_alt_rad = _angle_to_radians(moon_alt)
    moon_az_rad = _angle_to_radians(moon_az)
    moon_par_ang_rad = _angle_to_radians(moon_par_ang)
    
    moon_ra_deg = np.asarray(moon_pos.ra.to(u.deg).value, dtype=np.float64)
    moon_dec_deg = np.asarray(moon_pos.dec.to(u.deg).value, dtype=np.float64)
    
    sun_moon_ang_rad = _angle_to_radians(sun_moon_ang)
    
    lst_rad = _angle_to_radians(local_sidereal_times)
    
    return NightEventArrays(
        night_duration_minutes=num_slots,
        sunset=astropy_time_to_datetime(sunset_0),
        sunrise=astropy_time_to_datetime(sunrise_0),
        night_start=astropy_time_to_datetime(time_start),
        night_end=astropy_time_to_datetime(time_end),
        midnight=astropy_time_to_datetime(midnight_0),
        twilight_evening_12=astropy_time_to_datetime(twi_eve_0),
        twilight_morning_12=astropy_time_to_datetime(twi_mor_0),
        moonrise=astropy_time_to_datetime(moonrise_0) if moonrise_0 is not None else None,
        moonset=astropy_time_to_datetime(moonset_0) if moonset_0 is not None else None,
        moon_dist=moon_dist_au,
        sun_alt=pack_array(sun_alt_rad),
        sun_az=pack_array(sun_az_rad),
        sun_par_ang=pack_array(sun_par_ang_rad),
        moon_alt=pack_array(moon_alt_rad),
        moon_az=pack_array(moon_az_rad),
        moon_par_ang=pack_array(moon_par_ang_rad),
        moon_ra=pack_array(moon_ra_deg),
        moon_dec=pack_array(moon_dec_deg),
        sun_moon_ang=pack_array(sun_moon_ang_rad),
        local_sidereal_times=pack_array(lst_rad),
    )


def _round_minute(t: Time, up: bool = True) -> Time:
    """Round astropy Time to nearest minute."""
    dt = t.to_datetime(timezone=timezone.utc)
    if up:
        if dt.second > 0 or dt.microsecond > 0:
            dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
        else:
            dt = dt.replace(second=0, microsecond=0)
    else:
        dt = dt.replace(second=0, microsecond=0)
    return Time(dt)


def _angle_to_radians(angle) -> npt.NDArray[np.float64]:
    """Convert astropy Angle to radians numpy array."""
    return np.asarray(angle.to(u.rad).value, dtype=np.float64)
