# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import astropy.units as u
import numpy as np
import pytz
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
from lucupy.minimodel import Site, Variant, CloudCover, ImageQuality


class Env:
    @staticmethod
    def get_actual_conditions_variant(site: Site, # noqa
                                      times: Time) -> Optional[Variant]:
        """
        Return the weather variant.
        This should be site-based and time-based.
        """
        night_length = len(times)

        return Variant(
            start_time=times[0].to_datetime(pytz.UTC),
            iq=np.full(night_length, ImageQuality.IQ70),
            cc=np.full(night_length, CloudCover.CC50),
            wind_dir=Angle(np.full(night_length, 330.0), unit='deg'),
            wind_spd=Quantity(np.full(night_length, 5.0 * u.m / u.s))
        )
