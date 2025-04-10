# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from typing import final, Dict, TypeAlias, Tuple

import numpy.typing as npt
from astropy.coordinates import Angle, SkyCoord
from astropy.time import TimeDelta
from lucupy.decorators import immutable
from lucupy.minimodel import NightIndex, ObservationID, SkyBackground, TargetName


__all__ = [
    'TargetInfo',
    'TargetInfoMap',
    'TargetInfoNightIndexMap',
]

@final
@immutable
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
    max_dec: Angle
    min_dec: Angle
    alt: Angle
    az: Angle
    hourangle: Angle
    airmass: npt.NDArray[float]
    visibility_slot_idx: npt.NDArray[int]
    rem_visibility_frac: float


# Type aliases for TargetInfo information.
TargetInfoNightIndexMap: TypeAlias = Dict[NightIndex, TargetInfo]
TargetInfoMap: TypeAlias = Dict[Tuple[TargetName, ObservationID], TargetInfoNightIndexMap]
