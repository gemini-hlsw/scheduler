from collections.abc import Iterable
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.time import Time


def flatten(lst):
    """
    Flattens any iterable, no matter how irregular.
    Example: flatten([1, 2, [3, 4, 5], [[6, 7], 8, [9, 10]]])
    Deliberately left untyped to allow for maximum type usage.
    """
    for el in lst:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


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


def dmsstr2deg(s: str) -> float:
    if not s:
        raise ValueError(f'Illegal DMS string: {s}')

    sign = '+'
    if s[0] in SIGNS:
        sign = s[0]
        s = s[1:]

    result = s.split(':')
    if len(result) != 3:
        raise ValueError(f'Illegal DMS string: {s}')

    try:
        return dms2deg(int(result[0]), int(result[1]), float(result[2]), sign)
    except ValueError:
        import logging
        logging.error(f'Uhoh: {s}')
        raise


def dms2deg(d: int, m: int, s: float, sign: str) -> float:
    if sign not in SIGNS:
        raise ValueError(f'Illegal sign "{sign}" in DMS: {sign}{d}:{m}:{s}')
    dec = SIGNS[sign] * (d + m / 60.0 + s / 3600.0)
    return dec if dec < 180 else -(360 - dec)


def dms2rad(d: int, m: int, s: float, sign: str) -> float:
    return dms2deg(d, m, s, sign) * np.pi / 180.0


def hmsstr2deg(s: str) -> float:
    if not s:
        raise ValueError(f'Illegal HMS string: {s}')

    result = s.split(':')
    if len(result) != 3:
        raise ValueError(f'Illegal HMS string: {s}')

    return hms2deg(int(result[0]), int(result[1]), float(result[2]))


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
