from astropy.time import Time
import astropy.units as u
from typing import Optional


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
