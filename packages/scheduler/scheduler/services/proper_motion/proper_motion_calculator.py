# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import final, ClassVar, Dict, Final, Optional

from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np

from lucupy.minimodel import SiderealTarget
from lucupy.meta import Singleton

from scheduler.config import config

__all__ = [
    'ProperMotionCalculator',
]


@final
class ProperMotionCalculator(metaclass=Singleton):
    # Milliarcseconds per degree.
    _MAS_PER_DEGREE: Final[float] = 1000.0 * 3600.0

    # For a given epoch, determines the AstroPy Time at which the epoch began.
    _EPOCH2TIME: ClassVar[Dict[float, Time]] = {}

    # _DEFAULT_TIMESLOT_LENGTH: TimeDelta = TimeDelta(60.0 * u.second)
    _DEFAULT_TIMESLOT_LENGTH: TimeDelta = TimeDelta(config.collector.time_slot_length * u.min)

    def calculate_coordinates(self,
                              target: SiderealTarget,
                              start_time: Time,
                              num_time_slots: int,
                              time_slot_length: Optional[TimeDelta] = None) -> SkyCoord:
        if time_slot_length is None:
            time_slot_length = ProperMotionCalculator._DEFAULT_TIMESLOT_LENGTH

        # Convert to degrees/year
        pm_ra = target.pm_ra / ProperMotionCalculator._MAS_PER_DEGREE
        pm_dec = target.pm_dec / ProperMotionCalculator._MAS_PER_DEGREE
        epoch_time = ProperMotionCalculator._EPOCH2TIME.setdefault(target.epoch, Time(target.epoch, format='jyear'))

        # Create an array of times at once
        # times = start_time + time_slot_length * np.arange(num_time_slots)
        times = start_time + time_slot_length * int(num_time_slots/2)

        # Calculate the time offsets for all times at once
        time_offsets_years = (times - epoch_time).to(u.yr).value

        # Calculate new RA and Dec using NumPy broadcasting
        ras = target.ra + pm_ra * time_offsets_years
        decs = target.dec + pm_dec * time_offsets_years

        # Return a single SkyCoord object with all positions
        return SkyCoord(ra=ras * u.deg, dec=decs * u.deg, frame='icrs')
