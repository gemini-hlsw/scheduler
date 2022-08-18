from dataclasses import dataclass
from astropy.time import TimeDelta
from astropy.coordinates import Angle, SkyCoord
import numpy.typing as npt
from typing import Dict, Tuple
from lucupy.minimodel import NightIndex, ObservationID, SkyBackground, TargetName


@dataclass(frozen=True)
class TargetInfo:
    """
    Target information for a given target at a given site for a given night.
    * For a SiderealTarget, we have to account for proper motion, which is handled below.
    * For a NonsiderealTarget, we have to account for ephemeris data, which is handled in
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
TargetInfoNightIndexMap = Dict[NightIndex, TargetInfo]
TargetInfoMap = Dict[Tuple[TargetName, ObservationID], TargetInfoNightIndexMap]
