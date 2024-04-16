# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import final, ClassVar, Dict, Final, Optional

from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np

from lucupy.minimodel import SiderealTarget
from lucupy.meta import Singleton

__all__ = [
    'ProperMotionCalculator',
]


@final
class ProperMotionCalculator(metaclass=Singleton):
    # Milliarcseconds per degree.
    _MAS_PER_DEGREE: Final[float] = 1000.0 * 3600.0

    # For a given epoch, determines the AstroPy Time at which the epoch began.
    _EPOCH2TIME: ClassVar[Dict[float, Time]] = {}

    _DEFAULT_TIMESLOT_LENGTH: TimeDelta = TimeDelta(60.0 * u.second)

    def calculate_positions(self,
                            target: SiderealTarget,
                            start_time: Time,
                            num_time_slots: int,
                            time_slot_length: Optional[TimeDelta] = None) -> [SkyCoord]:
        if time_slot_length is None:
            time_slot_length = ProperMotionCalculator._DEFAULT_TIMESLOT_LENGTH

        pm_ra = target.pm_ra / ProperMotionCalculator._MAS_PER_DEGREE
        pm_dec = target.pm_dec / ProperMotionCalculator._MAS_PER_DEGREE
        epoch_time = ProperMotionCalculator._EPOCH2TIME.setdefault(target.epoch, Time(target.epoch, format='jyear'))

        ras = np.zeros(num_time_slots) * u.deg
        decs = np.zeros(num_time_slots) * u.deg

        for slot in range(num_time_slots):
            # Calculate each time incrementally
            current_time = start_time + time_slot_length * slot
            time_offsets = current_time - epoch_time
            # Calculate new RA and Dec
            ras[slot] = (target.ra + pm_ra * time_offsets.to(u.yr).value) * u.deg
            decs[slot] = (target.dec + pm_dec * time_offsets.to(u.yr).value) * u.deg

            # Return a single SkyCoord object with all positions
        return SkyCoord(ra=ras, dec=decs, frame='icrs')
