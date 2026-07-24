import numpy as np
import numpy.typing as npt
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from pydantic import BaseModel, ConfigDict, Field

from lucupy.minimodel import SkyBackground, Constraints  # noqa: F401 - SkyBackground re-exported via calculations.__init__
import lucupy.sky as sky

from scheduler.services.sight.calculations.arrays import unpack_array


import math
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
    elevation_min: float = Constraints.DEFAULT_AIRMASS_ELEVATION_MIN
    elevation_max: float = Constraints.DEFAULT_AIRMASS_ELEVATION_MAX
    
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
    moon_dist_bytes: bytes,
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
    moon_dist_m = unpack_array(moon_dist_bytes, n)  # meters, per-slot
    
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
        mask &= ((airmass >= Constraints.DEFAULT_AIRMASS_ELEVATION_MIN) &
                 (airmass <= Constraints.DEFAULT_AIRMASS_ELEVATION_MAX))
    
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
            moon_dist_m=moon_dist_m,
        )
        mask &= sky_brightness_arr <= constraints.target_sb
    
    # Step 4: Timing windows
    if constraints.timing_windows:
        timing_mask = np.zeros(n, dtype=bool)
        for tw in constraints.timing_windows:
            # Slot k is the instant night_start + k minutes. The legacy
            # calculator kept a slot iff window.start <= t_k <= window.end
            # (inclusive both ends): ceil on the start so a slot beginning
            # before the window opens is excluded, floor+1 on the end so the
            # last slot inside the window is kept.
            tw_start_min = _minute_index(tw.start, night_start, round_up=True)
            tw_end_min = _minute_index(tw.end, night_start, round_up=False) + 1

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


def _minute_index(dt: datetime, night_start: datetime, round_up: bool) -> int:
    """Convert datetime to a minute (slot) index from night start.

    round_up=True gives the first slot at or after ``dt`` (window start);
    round_up=False gives the last slot at or before ``dt`` (window end).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if night_start.tzinfo is None:
        night_start = night_start.replace(tzinfo=timezone.utc)

    delta_min = (dt - night_start).total_seconds() / 60.0
    return math.ceil(delta_min) if round_up else math.floor(delta_min)


def _calculate_sky_brightness_array(
    ra: npt.NDArray[np.float64],
    dec: npt.NDArray[np.float64],
    alt: npt.NDArray[np.float64],
    sun_alt: npt.NDArray[np.float64],
    moon_alt: npt.NDArray[np.float64],
    moon_ra: npt.NDArray[np.float64],
    moon_dec: npt.NDArray[np.float64],
    sun_moon_ang: npt.NDArray[np.float64],
    moon_dist_m: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate sky brightness (as a SkyBackground fraction) for each time slot.

    Vectorized the results so they can be applied further down the process.
    """
    n = len(ra)

    # Calculate target-moon angular separation
    target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    moon_coord = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)
    target_moon_ang = target_coord.separation(moon_coord)

    moon_phase = Angle(180.0 * u.deg - (sun_moon_ang * u.rad).to(u.deg))
    moon_zenith = Angle(90.0 * u.deg - (moon_alt * u.rad).to(u.deg))
    target_zenith = Angle(90.0 * u.deg - (alt * u.rad).to(u.deg))
    sun_zenith = Angle(90.0 * u.deg - (sun_alt * u.rad).to(u.deg))
    earth_moon_dist = moon_dist_m * u.m

    raw = sky.brightness.calculate_sky_brightness(
        moon_phase,
        target_moon_ang,
        earth_moon_dist,
        moon_zenith,
        target_zenith,
        sun_zenith,
    )
    return np.asarray(
        sky.brightness.convert_to_sky_background(raw), dtype=np.float64
    )
