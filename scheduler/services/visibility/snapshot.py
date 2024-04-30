from dataclasses import dataclass
from typing import Dict
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord, Angle
from astropy.time import TimeDelta
from typing import final

from lucupy.decorators import immutable
from lucupy.minimodel import SkyBackground


@final
@immutable
@dataclass(frozen=True)
class VisibilitySnapshot:
    visibility_slot_idx: npt.NDArray[int]
    visibility_time: TimeDelta

    @staticmethod
    def from_dict(ti_dict: Dict) -> 'VisibilitySnapshot':
        return VisibilitySnapshot(visibility_slot_idx=np.array(ti_dict['visibility_slot_idx'], dtype=int),
                                  visibility_time=TimeDelta(ti_dict['visibility_time']['value'],
                                                            format=ti_dict['visibility_time']['format']),
                                 )

    def to_dict(self) -> Dict:
        return {
            'visibility_slot_idx': self.visibility_slot_idx.tolist(),
            'visibility_time': {
                'value': self.visibility_time.sec,
                'format': self.visibility_time.format
            }
        }


@final
@immutable
@dataclass(frozen=True)
class TargetSnapshot:
    coord: SkyCoord
    alt: Angle
    az: Angle
    par_ang: Angle
    hourangle: Angle
    airmass: npt.NDArray[float]
    target_sb: SkyBackground
    sky_brightness: npt.NDArray[SkyBackground]