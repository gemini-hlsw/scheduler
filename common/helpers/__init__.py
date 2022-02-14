from astropy.time import Time
import astropy.units as u
import numpy as np
from typing import Optional


def round_minute(time: Time, up: bool = False) -> Time:
    """
    Round time down (truncate) or up to the nearest minute
    time: an astropy.Time object (can be an array)
    up: bool indicating whether to round up
    """
    t = time.copy()
    t.format = 'iso'
    t.out_subfmt = 'date_hm'

    arr = np.asarray(t)
    scalar_input = False
    if arr.ndim == 0:
        arr = arr[None]
        scalar_input = True

    if up:
        minute = 1.0 * u.min
        arr = [tm + minute if int(tm.strftime('%S')) else tm for tm in arr]

    if scalar_input:
        arr = np.squeeze(arr)
    return Time(arr.iso, format='iso', scale='utc')


def str_to_bool(s: Optional[str]) -> bool:
    """
    Returns true if and only if s is defined and some variant capitalization of 'yes' or 'true'.
    """
    return s is not None and s.strip().upper() in ['YES', 'TRUE']
