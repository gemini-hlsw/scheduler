from dataclasses import dataclass
from typing import Dict
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord, Angle
from astropy.time import TimeDelta
from typing import final

from lucupy.decorators import immutable
from lucupy.minimodel import SkyBackground

import itertools

def group_ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]


@final
@immutable
@dataclass(frozen=True)
class VisibilitySnapshot:
    """
    Visibility information needed to calculate the remaining visibility for
    each target.
    """
    visibility_slot_idx: npt.NDArray[int]
    visibility_time: TimeDelta

    @staticmethod
    def from_dict(ti_dict: Dict) -> 'VisibilitySnapshot':
        try:
            slot_list = [list(range(s[0], s[1] + 1)) for s in ti_dict['visibility_slot_idx']]
        except Exception as e:
            print(e)
            print(ti_dict['visibility_slot_idx'])
        return VisibilitySnapshot(visibility_slot_idx=np.array([x for xs in slot_list for x in xs], dtype=int),
                                  visibility_time=TimeDelta(ti_dict['visibility_time']['value'],
                                                            format=ti_dict['visibility_time']['format']),
                                 )

    def to_dict(self) -> Dict:
        visibility_ranges = group_ranges(self.visibility_slot_idx.tolist())
        return {
            'visibility_slot_idx': [*visibility_ranges],
            'visibility_time': {
                'value': self.visibility_time.sec,
                'format': self.visibility_time.format
            }
        }


@final
@immutable
@dataclass(frozen=True)
class TargetSnapshot:
    """
    Contains information of a target for a night.
    All values except target_sb are numpy arrays (shape == time_slots).
    """
    coord: SkyCoord
    alt: Angle
    az: Angle
    par_ang: Angle
    hourangle: Angle
    airmass: npt.NDArray[float]
    target_sb: SkyBackground
    sky_brightness: npt.NDArray[SkyBackground]