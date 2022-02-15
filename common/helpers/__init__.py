from astropy.time import Time
import astropy.units as u
import numpy as np

from typing import Optional


def round_minute(time: Time, up: bool = False) -> Time:
    """
    Round a time down (truncate) or up to the nearest minute
    time: an astropy.Time
    up: bool indicating whether to round up
    """
    t = time.copy()
    t.format = 'iso'
    t.out_subfmt = 'date_hm'
    if up:
        sec = t.strftime('%S').astype(int)
        idx = np.where(sec > 0)
        t[idx] += 1.0 * u.min
    return Time(t.iso, format='iso', scale='utc')


def str_to_bool(s: Optional[str]) -> bool:
    """
    Returns true if and only if s is defined and some variant capitalization of 'yes' or 'true'.
    """
    return s is not None and s.strip().upper() in ['YES', 'TRUE']


# A dict of signs for conversion.
SIGNS = {'': 1, '+': 1, '-': -1}


def dms2deg(d: int, m: int, s: float, sign: str) -> float:
    if sign not in SIGNS:
        raise ValueError(f"Illegal sign in DMS '{sign}{d}:{m}:{s}")

    return SIGNS[sign] * (d + m / 60.0 + s / 3600.0)


def dms2rad(d: int, m: int, s: float, sign: str) -> float:
    return dms2deg(d, m, s, sign) * np.pi / 180.0


def hms2deg(h: int, m: int, s: float) -> float:
    return h + m / 60.0 + s / 3600.0


def hms2rad(h: int, m: int, s: float) -> float:
    return hms2deg(h, m, s) * np.pi / 12.


def angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate the angular distance between two points on the sky.
    based on
    https://github.com/gemini-hlsw/lucuma-core/blob/master/modules/core/shared/src/main/scala/lucuma/core/math/Coordinates.scala#L52
    """
    phi_1 = dec1
    phi_2 = dec2
    delta_phi = dec2 - dec1
    delta_lambda = ra2 - ra1
    a = np.sin(delta_phi / 2)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2)**2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
