from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord, Angle
from astropy.time import TimeDelta
from typing import final

from lucupy.decorators import immutable
from lucupy.minimodel import SkyBackground
import itertools

__all__ = [
    'VisibilitySnapshot',
    'TargetSnapshot'
]


def group_ranges(i):
    for _, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]


@final
# @immutable
@dataclass(frozen=False)
class VisibilitySnapshot:
    """
    Visibility information needed to calculate the remaining visibility for
    each target.
    """
    visibility_slot_idx: npt.NDArray[int]
    visibility_time: TimeDelta

    @staticmethod
    def from_dict(ti_dict: Dict) -> 'VisibilitySnapshot':
        slot_list = [list(range(s[0], s[1] + 1)) for s in ti_dict['visibility_slot_idx']]
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

    @staticmethod
    def from_dict_days(ti_dict: Dict) -> Dict[str, 'VisibilitySnapshot']:
        """Create a target visibility dictionary by looping over days"""
        tv = None
        if ti_dict is not None:
            tv = {}
            for day in ti_dict.keys():
                tv[day] = VisibilitySnapshot.from_dict(ti_dict[day])
        return tv

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
    hourangle: Angle
    airmass: npt.NDArray[float]
    par_ang: Optional[Angle] = None
    target_sb: Optional[SkyBackground] = None
    sky_brightness: Optional[npt.NDArray[SkyBackground]] = None