import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
import astropy.units as u
from pydantic import BaseModel, ConfigDict, Field

from lucupy.minimodel import SkyBackground
import lucupy.sky as sky

from scheduler.services.sight.calculations.arrays import unpack_array


from datetime import datetime, timezone
from enum import Enum

class ElevationType(str, Enum):
    NONE = "none"
    HOUR_ANGLE = "hour_angle"
    AIRMASS = "airmass"


class TimingWindow(BaseModel):
    """Time window when observation can occur."""
    start: datetime
    end: datetime


class ObservationConstraints(BaseModel):
    """Observation constraints for Stage 2 calculation."""
    
    # Sky background
    target_sb: float = 1.0
    
    # Elevation
    elevation_type: ElevationType = ElevationType.AIRMASS
    elevation_min: float = 1.0
    elevation_max: float = 2.05
    
    # Timing windows
    timing_windows: list[TimingWindow] = Field(default_factory=list)
    
    # External flags
    has_resources: bool = True
    can_schedule: bool = True


class Stage2Result(BaseModel):
    """Result of Stage 2 visibility calculation."""
    visibility_mask: list[bool]
    remaining_minutes: int
    sky_brightness: list[float] | None = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


def calculate_visibility(
    # Stage 1 data (packed bytes)
    alt_bytes: bytes,
    az_bytes: bytes,
    airmass_bytes: bytes,
    hourangle_bytes: bytes,
    ra_bytes: bytes,
    dec_bytes: bytes,
    # Night event data (packed bytes)
    sun_alt_bytes: bytes,
    moon_alt_bytes: bytes,
    moon_ra_bytes: bytes,
    moon_dec_bytes: bytes,
    sun_moon_ang_bytes: bytes,
    moon_dist: float,
    # Night timing
    night_start: datetime,
    night_duration_minutes: int,
    # Constraints
    constraints: ObservationConstraints,
) -> Stage2Result:
    """
    Calculate visibility mask based on Stage 1 data and observation constraints.
    """
    n = night_duration_minutes
    
    # Early exit if resources unavailable or can't schedule
    if not constraints.has_resources or not constraints.can_schedule:
        return Stage2Result(
            visibility_mask=[False] * n,
            remaining_minutes=0,
            sky_brightness=None,
        )
    
    # Unpack arrays
    alt = unpack_array(alt_bytes, n)  # radians
    airmass = unpack_array(airmass_bytes, n)
    hourangle = unpack_array(hourangle_bytes, n)  # radians
    ra = unpack_array(ra_bytes, n)
    dec = unpack_array(dec_bytes, n)
    sun_alt = unpack_array(sun_alt_bytes, n)  # radians
    moon_alt = unpack_array(moon_alt_bytes, n)
    moon_ra = unpack_array(moon_ra_bytes, n)
    moon_dec = unpack_array(moon_dec_bytes, n)
    sun_moon_ang = unpack_array(sun_moon_ang_bytes, n)
    
    # Step 1: Sun altitude filter (astronomical twilight, sun < -12°)
    sun_alt_deg = np.degrees(sun_alt)
    mask = sun_alt_deg <= -12.0
    
    # Step 2: Elevation constraints
    if constraints.elevation_type == ElevationType.AIRMASS:
        mask &= (airmass >= constraints.elevation_min) & (airmass <= constraints.elevation_max)
    elif constraints.elevation_type == ElevationType.HOUR_ANGLE:
        hourangle_deg = np.degrees(hourangle)
        mask &= (hourangle_deg >= constraints.elevation_min) & (hourangle_deg <= constraints.elevation_max)
    # NONE: use default airmass
    elif constraints.elevation_type == ElevationType.NONE:
        mask &= (airmass >= 1.0) & (airmass <= 2.05)
    
    # Step 3: Sky brightness constraint
    sky_brightness_arr = None
    if constraints.target_sb < 1.0:
        sky_brightness_arr = _calculate_sky_brightness_array(
            ra=ra,
            dec=dec,
            alt=alt,
            sun_alt=sun_alt,
            moon_alt=moon_alt,
            moon_ra=moon_ra,
            moon_dec=moon_dec,
            sun_moon_ang=sun_moon_ang,
            moon_dist=moon_dist,
        )
        mask &= sky_brightness_arr <= constraints.target_sb
    
    # Step 4: Timing windows
    if constraints.timing_windows:
        timing_mask = np.zeros(n, dtype=bool)
        for tw in constraints.timing_windows:
            # Convert timing window to minute indices
            tw_start_min = _datetime_to_minute_index(tw.start, night_start)
            tw_end_min = _datetime_to_minute_index(tw.end, night_start)
            
            # Clamp to valid range
            tw_start_min = max(0, tw_start_min)
            tw_end_min = min(n, tw_end_min)
            
            if tw_start_min < tw_end_min:
                timing_mask[tw_start_min:tw_end_min] = True
        
        mask &= timing_mask
    
    remaining_minutes = int(np.sum(mask))
    
    return Stage2Result(
        visibility_mask=mask.tolist(),
        remaining_minutes=remaining_minutes,
        sky_brightness=sky_brightness_arr.tolist() if sky_brightness_arr is not None else None,
    )


def _datetime_to_minute_index(dt: datetime, night_start: datetime) -> int:
    """Convert datetime to minute index from night start."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if night_start.tzinfo is None:
        night_start = night_start.replace(tzinfo=timezone.utc)
    
    delta = dt - night_start
    return int(delta.total_seconds() / 60)


def _calculate_sky_brightness_array(
    ra: npt.NDArray[np.float64],
    dec: npt.NDArray[np.float64],
    alt: npt.NDArray[np.float64],
    sun_alt: npt.NDArray[np.float64],
    moon_alt: npt.NDArray[np.float64],
    moon_ra: npt.NDArray[np.float64],
    moon_dec: npt.NDArray[np.float64],
    sun_moon_ang: npt.NDArray[np.float64],
    moon_dist: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate sky brightness for each time slot.
    """
    n = len(ra)
    
    # Calculate target-moon angular separation
    target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    moon_coord = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)
    target_moon_ang = target_coord.separation(moon_coord)
    
    # Convert to required units
    moon_phase = 180.0 * u.deg - (sun_moon_ang * u.rad).to(u.deg)
    moon_zenith = 90.0 * u.deg - (moon_alt * u.rad).to(u.deg)
    target_zenith = 90.0 * u.deg - (alt * u.rad).to(u.deg)
    sun_zenith = 90.0 * u.deg - (sun_alt * u.rad).to(u.deg)
    
    brightness = np.full(n, SkyBackground.SBANY, dtype=np.float64)
    
    for i in range(n):
        try:
            raw = sky.brightness.calculate_sky_brightness(
                moon_phase[i],
                target_moon_ang[i],
                moon_dist,
                moon_zenith[i],
                target_zenith[i],
                sun_zenith[i],
            )
            brightness[i] = sky.brightness.convert_to_sky_background(raw)
        except Exception:
            brightness[i] = SkyBackground.SBANY
    
    return brightness
