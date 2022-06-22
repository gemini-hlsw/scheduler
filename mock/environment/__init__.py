from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity

from common.minimodel import Site, Variant, CloudCover, ImageQuality, WaterVapor


class Env:
    def get_actual_conditions_variant(self,
                                      site: Site,
                                      times: Time) -> Optional[Variant]:
        """
        Return the weather variant.
        This should be site-based and time-based.
        """
        night_length = len(times)

        return Variant(
            iq=np.full(night_length, ImageQuality.IQ70),
            cc=np.full(night_length, CloudCover.CC50),
            wv=np.full(night_length, WaterVapor.WVANY),
            wind_dir=Angle(np.full(night_length, 330.0), unit='deg'),
            wind_sep=Angle(np.full(night_length, 40.0), unit='deg'),
            wind_spd=Quantity(np.full(night_length, 5.0 * u.m / u.s))
        )
