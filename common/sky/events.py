from typing import Tuple, Union

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time
from pytz import timezone

from common.sky.constants import EQUAT_RAD
from common.sky.moon import Moon
from common.sky.sun import Sun
from common.sky.utils import local_midnight_time


def night_events(time: Time, location: EarthLocation, localtzone: timezone) -> \
        Tuple[Time, Union[npt.NDArray[float], Time],
              Union[npt.NDArray[float], Time],
              Union[npt.NDArray[float], Time],
              Union[npt.NDArray[float], Time],
              Union[npt.NDArray[float], Time],
              Union[npt.NDArray[float], Time]]:
    """
    Compute phenomena for a given night.

    This is mostly a testbed that prints results directly.

    Parameters
    ----------
    time : astropy Time, single or array
        input time; if before noon, events of previous night are computed.
    location : EarthLocation
    localtzone : timezone object.
    """
    # prototype for the events of a single night -- sunset and rise,
    # twilights, and moonrise and set.

    time = Time(np.asarray(time.iso), format='iso')
    scalar_input = False
    if time.ndim == 0:
        time = time[None]  # Makes 1D
        scalar_input = True

    nt = len(time)

    midnight = local_midnight_time(time, localtzone)  # nearest clock-time midnight (UT)
    # allow separate rise and set altitudes for horizon effects
    horiz = (-0.883 - np.sqrt(2. * location.height / EQUAT_RAD) * (180. / np.pi)) * u.deg

    set_alt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    rise_alt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin

    # Sun
    sunrise, sunset, even_12twi, morn_12twi = Sun.rise_and_set(location, time, midnight, set_alt, rise_alt)

    # Moon
    moonrise, moonset = Moon().rise_and_set(location, midnight, set_alt, rise_alt)

    if scalar_input:
        sunset = np.squeeze(sunset)
        sunrise = np.squeeze(sunrise)
        even_12twi = np.squeeze(even_12twi)
        morn_12twi = np.squeeze(morn_12twi)
        moonrise = np.squeeze(moonrise)
        moonset = np.squeeze(moonset)

    return midnight, sunset, sunrise, even_12twi, morn_12twi, moonrise, moonset
