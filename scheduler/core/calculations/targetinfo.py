# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from typing import final, Dict, TypeAlias, Tuple

import numpy as np
import numpy.typing as npt
import astropy.units as u
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
    coord: SkyCoord
    alt: Angle
    az: Angle
    par_ang: Angle
    hourangle: Angle
    airmass: npt.NDArray[float]
    sky_brightness: npt.NDArray[SkyBackground]
    visibility_slot_idx: npt.NDArray[int]
    visibility_slot_filter: npt.NDArray[int]
    visibility_time: TimeDelta
    rem_visibility_time: TimeDelta
    rem_visibility_frac: float

    def mean_airmass(self, interval: npt.NDArray[int]):
        return np.mean(self.airmass[interval])

    @staticmethod
    def from_dict(ti_dict: Dict) -> 'TargetInfo':
        return TargetInfo(coord=SkyCoord(ra=np.array(ti_dict['coord']['ra']) * u.deg,
                                         dec=np.array(ti_dict['coord']['dec']) * u.deg,
                                         frame=ti_dict['coord']['frame']),
                          alt=Angle(np.array(ti_dict['alt']['value']), unit=ti_dict['alt']['unit']),
                          az=Angle(np.array(ti_dict['az']['value']), unit=ti_dict['az']['unit']),
                          par_ang=Angle(np.array(ti_dict['par_ang']['value']), unit=ti_dict['par_ang']['unit']),
                          hourangle=Angle(np.array(ti_dict['hourangle']['value']), unit=ti_dict['hourangle']['unit']),
                          airmass=np.array(ti_dict['airmass']),
                          sky_brightness=np.array(ti_dict['sky_brightness']),
                          visibility_slot_idx=np.array(ti_dict['visibility_slot_idx']),
                          visibility_slot_filter=np.array(ti_dict['visibility_slot_filter']),
                          visibility_time=TimeDelta(ti_dict['visibility_time']['value'],
                                                    format=ti_dict['visibility_time']['format']),
                          rem_visibility_time=TimeDelta(ti_dict['rem_visibility_time']['value'],
                                                        format=ti_dict['rem_visibility_time']['format']),
                          rem_visibility_frac=ti_dict['rem_visibility_frac']
                          )

    def to_dict(self) -> Dict:
        return {
            'coord': {
                'ra': self.coord.ra.deg.tolist(),
                'dec': self.coord.dec.deg.tolist(),
                'frame': self.coord.frame.name
            },
            'alt': {
                'value': self.alt.value.tolist(),
                'unit': str(self.alt.unit)
            },
            'az': {
                'value': self.az.value.tolist(),
                'unit': str(self.az.unit)
            },
            'par_ang': {
                'value': self.par_ang.value.tolist(),
                'unit': str(self.par_ang.unit)
            },
            'hourangle': {
                'value': self.hourangle.value.tolist(),
                'unit': str(self.hourangle.unit)
            },
            'airmass': self.airmass.tolist(),
            'sky_brightness': self.sky_brightness.tolist(),
            'visibility_slot_idx': self.visibility_slot_idx.tolist(),
            'visibility_slot_filter': self.visibility_slot_filter.tolist(),
            'visibility_time': {
                'value': self.visibility_time.sec,
                'format': self.visibility_time.format
            },
            'rem_visibility_time': {
                'value': self.rem_visibility_time.sec,
                'format': self.rem_visibility_time.format
            },
            'rem_visibility_frac': self.rem_visibility_frac,
        }


# Type aliases for TargetInfo information.
TargetInfoNightIndexMap: TypeAlias = Dict[NightIndex, TargetInfo]
TargetInfoMap: TypeAlias = Dict[Tuple[TargetName, ObservationID], TargetInfoNightIndexMap]
