from collections.abc import Iterable
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.time import Time
import re


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


barcodes = {'0.25arcsec': '10005371',
            '0.5arcsec': '10005372',
            '0.75arcsec': '10005373',
            '1.0arcsec': '10005374',
            '1.5arcsec': '10005375',
            '2.0arcsec': '10005376',
            '5.0arcsec': '10005377',
            'IFU-2': '10000007',
            'IFU-B': '10000008',
            'IFU-R': '10000009',
            'IFU-NS-2': '10000010',
            'IFU-NS-B': '10000011',
            'IFU-NS-R': '10000012',
            'NS0.5arcsec': '10005388',
            'NS0.75arcsec': '10005389',
            'NS1.0arcsec': '10005390',
            'NS1.5arcsec': '10005391',
            'NS2.0arcsec': '10005392',
            'PinholeC': '10005381',
            'focus_array_new': '10000005'}

inst_decode = {'GMOS': '1', 'F2': '3'}
sem_decode = {'A': '0', 'B': '1'}
prog_decode = {'Q': '0', 'C': '1', 'L': '2', 'F': '3', 'S': '8', 'D': '9'}


decoder = {'A': '0', 'B': '1', 'Q': '0',
           'C': '1', 'LP': '2', 'FT': '3',
           'SV': '8', 'DD': '9'}
bar_decoder = {value: key for key, value in decoder.items()}
pattern = '|'.join(decoder.keys())
sem_code = {value: key for key, value in sem_decode.items()}


def mask_to_barcode(mask: str, inst: Optional[str]) -> str:
    """
    Convert a mask string to a barcode string.
    """
    if mask in barcodes:
        return barcodes[mask]
    else:
        # return inst_decode[inst] + sem_decode[mask[6]] + prog_decode[mask[7]] + mask[-6:-3] + mask[-2:]
        return inst_decode[inst] + re.sub(f'({pattern})', lambda m: decoder[m.group()], mask).replace('-', '')[6:]


def barcode_to_mask(barcode: str, rootname: Optional[str]) -> str:
    """
    Convert a barcode string to a mask string.
    """
    
    if barcode in barcodes.values():
        return list(barcodes.keys())[list(barcodes.values()).index(barcode)]
    else:
        # Root is like 'GS2022'
        return rootname + sem_code[barcode[1]] + bar_decoder[barcode[2]] + barcode[3:6] + '-' + barcode[6:]
