from astropy.time import Time
import astropy.units as u
from typing import Optional
import numpy as np

def round_min(time: Time, up: bool = False) -> Time:
    """
    Round a time down (truncate) or up to the nearest minute
    time: an astropy.Time
    up: bool indicating whether to round up
    """
    t = time.copy()
    t.format = 'iso'
    t.out_subfmt = 'date_hm'
    if up:
        sec = int(t.strftime('%S'))
        if sec:
            t += 1.0 * u.min
    return Time(t.iso, format='iso', scale='utc')


def str_to_bool(s: Optional[str]) -> bool:
    """
    Returns true if and only if s is defined and some variant capitalization of 'yes' or 'true'.
    """
    return s is not None and s.strip().upper() in ['YES', 'TRUE']


  
def dms2deg(dms):

    if dms is None:
        return None

    d = float(dms[0])
    m = float(dms[1])
    s = float(dms[2])
    sign = dms[3]
    dd = d + m / 60. + s / 3600.

    if sign == '-':
        dd *= -1.

    return dd


def dms2rad(dms):
    if dms is None:
        return None
    
    dd = dms2deg(dms)
    rad = dd * np.pi / 180.
    return rad


def hms2rad(hms):
    if hms is None:
        return None

    h = float(hms[0])
    m = float(hms[1])
    s = float(hms[2])
    hours = h + m / 60. + s / 3600.
    rad = hours * np.pi / 12.

    return rad

def angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate the angular distance between two points on the sky.
    based on https://github.com/gemini-hlsw/lucuma-core/blob/master/modules/core/shared/src/main/scala/lucuma/core/math/Coordinates.scala#L52
    """
    φ1 = dec1
    φ2 = dec2
    delta_φ = dec2 - dec1
    delta_λ = ra2 - ra1
    a = np.sin(delta_φ / 2)**2 + np.cos(φ1) * np.cos(φ2) * np.sin(delta_λ / 2)**2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

