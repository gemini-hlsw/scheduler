from typing import Final
import astropy.units as u
from astropy.time import Time

"""
Constants that are used here and there.  Some are Quantities,
others are just floats. Not all are used.

The planet-coefs are for series expansions for the phase functions
of the planets, used in predicting apparent magnitude.  See code.
"""

PI: Final = 3.14159265358979
TWOPI: Final = 6.28318530717959
PI_OVER_2: Final = 1.57079632679490  # /* From Abramowitz & Stegun */
ARCSEC_IN_RADIAN: Final = 206264.8062471
DEG_IN_RADIAN: Final = 57.2957795130823
HRS_IN_RADIAN: Final = 3.819718634205
KMS_AUDAY: Final = 1731.45683633  # /* km per sec in 1 AU/day */
SPEED_OF_LIGHT: Final = 299792.458  # /* in km per sec ... exact. */
SS_MASS: Final = 1.00134198  # /* solar system mass in solar units */
J2000: Final = 2451545.  # /* Julian date at standard epoch */
J2000_Time: Time = Time(2451545., format='jd')  # J2000 rendered as a Time
SEC_IN_DAY: Final = 86400.
FLATTEN: Final = 0.003352813  # /* flattening of earth, 1/298.257 */
EQUAT_RAD: Final = 6378137. * u.m  # /* equatorial radius of earth, meters */
EARTHRAD_IN_AU: Final = 23454.7910556298  # /* number of earth rad in 1 au */
ASTRO_UNIT: Final = 1.4959787066e11  # /* 1 AU in meters */
RSUN: Final = 6.96000e8  # /* IAU 1976 recom. solar radius, meters */
RMOON: Final = 1.738e6  # /* IAU 1976 recom. lunar radius, meters */
PLANET_TOL: Final = 3.  # /* flag if nearer than 3 degrees
KZEN: Final = 0.172  # V-band zenith extinction for sky-brightness
