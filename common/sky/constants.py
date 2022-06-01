from typing import Final

import astropy.units as u
from astropy.coordinates.distances import Distance
from astropy.time import Time

"""
Constants that are used here and there.  Some are Quantities,
others are just floats. Not all are used.

The planet-coefs are for series expansions for the phase functions
of the planets, used in predicting apparent magnitude. See code.
"""

PI: Final[float] = 3.14159265358979
TWOPI: Final[float] = 6.28318530717959
PI_OVER_2: Final[float] = 1.57079632679490
ARCSEC_IN_RADIAN: Final[float] = 206264.8062471
DEG_IN_RADIAN: Final[float] = 57.2957795130823
HRS_IN_RADIAN: Final[float] = 3.819718634205
KMS_AUDAY: Final[float] = 1731.45683633  # /* km per sec in 1 AU/day */
SPEED_OF_LIGHT: Final[float] = 299792.458  # /* in km per sec ... exact. */
SS_MASS: Final[float] = 1.00134198  # /* solar system mass in solar units */
J2000: Final[float] = 2451545.  # /* Julian date at standard epoch */
J2000_Time: Final[Time] = Time(2451545., format='jd')  # J2000 rendered as a Time
SEC_IN_DAY: Final[float] = 86400.
FLATTEN: Final[float] = 0.003352813  # /* flattening of earth, 1/298.257 */
EQUAT_RAD: Final[Distance] = 6378137. * u.m  # /* equatorial radius of earth, meters */
EARTHRAD_IN_AU: Final[float] = 23454.7910556298  # /* number of earth rad in 1 au */
ASTRO_UNIT: Final[float] = 1.4959787066e11  # /* 1 AU in meters */
RSUN: Final[float] = 6.96000e8  # /* IAU 1976 recom. solar radius, meters */
RMOON: Final[float] = 1.738e6  # /* IAU 1976 recom. lunar radius, meters */
PLANET_TOL: Final[float] = 3.  # /* flag if nearer than 3 degrees
KZEN: Final[float] = 0.172  # mag / airmass relation for Hale Pohaku
