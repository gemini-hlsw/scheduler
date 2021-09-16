from astropy.time import Time
import astropy.units as u
import numpy as np

def roundMin(time: Time, up=False) -> Time:
    """
    Round a time down (truncate) or up to the nearest minute
    time : astropy.Time
    up: bool   Round up?s
    """
    

    t = time.copy()
    t.format = 'iso'
    t.out_subfmt = 'date_hm'
    if up:
        sec = int(t.strftime('%S'))
        if sec != 0:
            t += 1.0*u.min
    return Time(t.iso, format='iso', scale='utc')