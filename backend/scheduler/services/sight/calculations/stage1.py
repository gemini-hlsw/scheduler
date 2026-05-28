import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from pydantic import BaseModel

import lucupy.sky as sky
from lucupy.minimodel import TargetTag

from scheduler.services.sight.calculations.arrays import pack_array, unpack_array
from scheduler.services.sight.calculations.night_events import site_to_earth_location

if TYPE_CHECKING:
    from scheduler.services.sight.database.models import Site, Target, NightEvent


_TAG_STR_TO_ENUM = {
    'majorbody': TargetTag.MAJORBODY,
    'asteroid':  TargetTag.ASTEROID,
    'comet':     TargetTag.COMET,
}


class Stage1Arrays(BaseModel):
    """
    Stage 1 calculation results ready for database storage.
    
    Coordinates (ra, dec) are in degrees.
    Angles (alt, az, hourangle, par_ang) are in radians.
    """
    night_duration_minutes: int
    
    # Packed arrays (bytes)
    ra: bytes       # degrees
    dec: bytes      # degrees
    alt: bytes      # radians
    az: bytes       # radians
    hourangle: bytes  # radians
    airmass: bytes
    par_ang: bytes | None  # radians
    
    class Config:
        arbitrary_types_allowed = True


def calculate_stage1(
    target: "Target",
    site: "Site",
    night_event: "NightEvent",
    time_slot_length_minutes: int = 1,
) -> Stage1Arrays:
    """
    Calculate Stage 1 data for a target on a given night.
    
    Args:
        target: Target model
        site: Site model
        night_event: NightEvent model with precomputed night data
        time_slot_length_minutes: Time slot length in minutes
        
    Returns:
        Stage1Arrays ready for database storage
    """
    num_time_slots = night_event.night_duration_minutes
    location = site_to_earth_location(site)
    time_slot_length = TimeDelta(time_slot_length_minutes * 60, format='sec')
    
    # Build time grid for this night
    night_start = Time(night_event.night_start)
    time_grid = night_start + np.arange(num_time_slots) * time_slot_length
    
    # Get coordinates based on target type
    if target.is_sidereal:
        coord = _calculate_sidereal_coordinates(
            target=target,
            start_time=night_start,
            num_time_slots=num_time_slots,
            time_slot_length=time_slot_length,
        )
    else:
        coord = _calculate_nonsidereal_coordinates(
            target=target,
            site=site,
            night_event=night_event,
            num_time_slots=num_time_slots,
            time_slot_length_minutes=time_slot_length_minutes,
        )
    
    # Unpack local sidereal times from night_event
    lst_rad = unpack_array(night_event.local_sidereal_times, num_time_slots)
    lst = Angle(lst_rad * u.rad)
    
    # Calculate hour angle
    hourangle = lst - coord.ra
    hourangle.wrap_at(12.0 * u.hour, inplace=True)
    
    # Calculate altitude, azimuth, parallactic angle
    alt, az, par_ang = sky.Altitude.above(coord.dec, hourangle, location.lat)
    
    # Calculate airmass
    airmass = sky.true_airmass(alt)
    
    # Convert to numpy arrays
    ra_val = coord.ra.to(u.deg).value
    dec_val = coord.dec.to(u.deg).value

    if np.isscalar(ra_val):
        ra_deg = np.full(num_time_slots, ra_val, dtype=np.float64)
        dec_deg = np.full(num_time_slots, dec_val, dtype=np.float64)
    else:
        ra_deg = np.asarray(ra_val, dtype=np.float64)
        dec_deg = np.asarray(dec_val, dtype=np.float64)
    
    # Already arrays from sky.Altitude.above broadcasting
    alt_rad = np.asarray(alt.to(u.rad).value, dtype=np.float64)
    az_rad = np.asarray(az.to(u.rad).value, dtype=np.float64)
    ha_rad = np.asarray(hourangle.to(u.rad).value, dtype=np.float64)
    airmass_arr = np.asarray(airmass, dtype=np.float64)
    
    par_ang_bytes = None
    if par_ang is not None:
        par_ang_rad = np.asarray(par_ang.to(u.rad).value, dtype=np.float64)
        par_ang_bytes = pack_array(par_ang_rad)
    
    return Stage1Arrays(
        night_duration_minutes=num_time_slots,
        ra=pack_array(ra_deg),
        dec=pack_array(dec_deg),
        alt=pack_array(alt_rad),
        az=pack_array(az_rad),
        hourangle=pack_array(ha_rad),
        airmass=pack_array(airmass_arr),
        par_ang=par_ang_bytes,
    )


# =============================================================================
# Sidereal Target Calculations
# =============================================================================

# Milliarcseconds per degree
_MAS_PER_DEGREE: float = 1000.0 * 3600.0

# Cache for epoch times
_EPOCH2TIME: dict[float, Time] = {}


def _calculate_sidereal_coordinates(
    target: "Target",
    start_time: Time,
    num_time_slots: int,
    time_slot_length: TimeDelta,
) -> SkyCoord:
    """
    Calculate coordinates for a sidereal target with proper motion correction.
    
    For sidereal targets, proper motion over a single night is negligible,
    so we calculate at the midpoint and broadcast to all time slots.
    """
    pm_ra = (target.pm_ra or 0.0) / _MAS_PER_DEGREE
    pm_dec = (target.pm_dec or 0.0) / _MAS_PER_DEGREE
    
    epoch = target.epoch or 2000.0
    epoch_time = _EPOCH2TIME.setdefault(epoch, Time(epoch, format='jyear'))
    
    # Calculate at midpoint of night (matches original implementation)
    mid_time = start_time + time_slot_length * int(num_time_slots / 2)
    
    # Time offset in years
    time_offset_years = (mid_time - epoch_time).to(u.yr).value
    
    # Apply proper motion
    ras = target.base_ra + pm_ra * time_offset_years
    decs = target.base_dec + pm_dec * time_offset_years
    
    return SkyCoord(ra=ras * u.deg, dec=decs * u.deg, frame='icrs')


# =============================================================================
# Non-sidereal Target Calculations
# =============================================================================

def _calculate_nonsidereal_coordinates(
    target: "Target",
    site: "Site",
    night_event: "NightEvent",
    num_time_slots: int,
    time_slot_length_minutes: int,
) -> SkyCoord:
    """
    Calculate coordinates for a non-sidereal target using Horizons.
    
    Based on EphemerisCalculator logic.
    """
    # Import here to avoid circular imports and allow horizons to be optional
    from scheduler.services.horizons import horizons_session
    
    sunset = Time(night_event.sunset)
    sunrise = Time(night_event.sunrise)
    twilight_evening_12 = Time(night_event.twilight_evening_12)
    
    # Create site adapter for horizons
    site_adapter = _HorizonsSiteAdapter(site)
    
    # Create target adapter for horizons
    target_adapter = _HorizonsTargetAdapter(target)
    
    # Query Horizons (1-minute resolution from sunset to sunrise)
    with horizons_session(
        site_adapter,
        sunset.to_datetime(),
        sunrise.to_datetime(),
        time_slot_length=1,
    ) as hs:
        ephemerides = hs.get_ephemerides(target_adapter)
        coords = ephemerides.coordinates
        
        # Extract RA/Dec arrays (Horizons returns radians)
        ras = np.array([c.ra for c in coords])
        decs = np.array([c.dec for c in coords])
        
        # Create SkyCoord from Horizons data (in radians)
        eph_coord = SkyCoord(ra=ras * u.rad, dec=decs * u.rad, frame='icrs')
    
    # Trim coordinates to desired subset
    # Calculate offset from sunset to twilight
    sunset_to_twi = (twilight_evening_12 - sunset).to_datetime()
    start_time_slot = int(sunset_to_twi.total_seconds() / 60)
    end_time_slot = start_time_slot + num_time_slots * time_slot_length_minutes
    
    # Resample if time slot length is not 1 minute
    coord = eph_coord[start_time_slot:end_time_slot:time_slot_length_minutes]
    
    return coord


class _HorizonsSiteAdapter:
    """Adapter to match horizons_session expected site interface.
    Sight uses "Gemini North_*", while the existing files are "GN_*"
    """

    # Horizons observatory codes, keyed by sight DB Site.name.
    _COORDINATE_CENTERS = {
        "Gemini North": "568@399",
        "Gemini South": "I11@399",
    }

    # Sight DB Site.name -> lucupy Site enum short name.
    _DB_NAME_TO_SHORT = {
        "Gemini North": "GN",
        "Gemini South": "GS",
    }

    def __init__(self, site: "Site"):
        self.name = self._DB_NAME_TO_SHORT.get(site.name, site.name)
        self.location = site_to_earth_location(site)
        self.coordinate_center = self._COORDINATE_CENTERS.get(site.name, "500@399")


class _HorizonsTargetAdapter:
    """Adapter to match horizons get_ephemerides expected target interface."""

    def __init__(self, target: "Target"):
        self.name = target.name
        self.horizons_id = target.horizons_id
        self.des = target.horizons_id  # designation used by horizons
        tag_str = (target.tag or '').lower()
        self.tag = _TAG_STR_TO_ENUM.get(tag_str)
        if self.tag is None and tag_str:
            logging.getLogger(__name__).warning(
                f'Unknown target tag {tag_str!r} for {target.name}; '
                'falling back to DES= branch in horizons client.'
            )