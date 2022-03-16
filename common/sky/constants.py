"""
Constants that are used here and there.  Some are Quantities,
others are just floats. Not all are used.

The planet-coefs are for series expansions for the phase functions
of the planets, used in predicting apparent magnitude.  See code.
"""

PI = 3.14159265358979
TWOPI = 6.28318530717959
PI_OVER_2 = 1.57079632679490  # /* From Abramowitz & Stegun */
ARCSEC_IN_RADIAN = 206264.8062471
DEG_IN_RADIAN = 57.2957795130823
HRS_IN_RADIAN = 3.819718634205
KMS_AUDAY = 1731.45683633  # /* km per sec in 1 AU/day */
SPEED_OF_LIGHT = 299792.458  # /* in km per sec ... exact. */
SS_MASS = 1.00134198  # /* solar system mass in solar units */
J2000 = 2451545.  # /* Julian date at standard epoch */
J2000_Time = Time(2451545., format='jd')  # J2000 rendered as a Time
SEC_IN_DAY = 86400.
FLATTEN = 0.003352813  # /* flattening of earth, 1/298.257 */
EQUAT_RAD = 6378137. * u.m  # /* equatorial radius of earth, meters */
EARTHRAD_IN_AU = 23454.7910556298  # /* number of earth rad in 1 au */
ASTRO_UNIT = 1.4959787066e11  # /* 1 AU in meters */
RSUN = 6.96000e8  # /* IAU 1976 recom. solar radius, meters */
RMOON = 1.738e6  # /* IAU 1976 recom. lunar radius, meters */
PLANET_TOL = 3.  # /* flag if nearer than 3 degrees
KZEN = 0.172  # V-band zenith extinction for sky-brightness

ALT15 = 41.7291 * u.deg  # altitude at which true airm = 1.5
ALT20 = 29.8796 * u.deg  # same for 2.0
ALT30 = 19.278 * u.deg  # and 3.0

SIDRATE = 1.0027379093  # ratio of sidereal to solar rate
SIDDAY = TimeDelta(1., format='jd') / 1.0027379093
ZERO_TIMEDELTA = TimeDelta(0., format='jd')

# list planets so dictionary entries can be called up in order.
PLANETNAMES = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
# Phase brightness coefficients for the inner planets.
PLANETPHASECOEFS = {'mercury': (0.013363763741181076, -0.2480840022313796,
                                1.6325515091649714, -4.9390499605838665, 7.718379797341275, -6.131445146202686,
                                3.7914559630732065, -0.616),
                    'venus': (
                    0.09632276402543158, -0.5292390263170846, 1.2103116107350298, -0.05981450198047742, -4.38394),
                    'mars': (-0.4274213867715291, 1.2988953215615762, -1.601),
                    'jupiter': (-9.40), 'saturn': (-9.22), 'uranus': (-7.19), 'neptune': (-6.87)}
# These are adapted (converted to radian argument) from expressions given
# by Mallama, A. Wang, D., and Howard, R. A., Icarus 155, 253 (2002) for Mercury,
# Mallama, A. Wang, D., and Howard, R. A. Icarus 182, 10 (2005) for Venus, and
# Mallama, A., Icarus 192, 404 (2007) for Mars.  For the outer planets the phase angle
# is always nearly zero, so no phase angle correction is applied to Jupiter and further
# planets -- their magnitudes are adjusted only for sun-planet and earth-planet inverse square
# dimming. No attempt is made to account for the varying aspect of Saturn's rings.