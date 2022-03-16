from threading import local
from typing import Tuple, Union, Optional
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import GeocentricTrueEcliptic
from astropy.coordinates import SkyCoord, PrecessedGeocentric, Angle, Longitude, Distance
import numpy as np


class _Constants:
    """
    
    Constants that are used here and there.  Some are Quantities,
    others are just floats. Not all are used.

    The planet-coefs are for series expansions for the phase functions
    of the planets, used in predicting apparent magnitude.  See code."""

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

class _MoonLocationVariables:
    """
    Helper variables for the moon location method calculations.
    """
    def __init__(self, time):
        self.time = time
        self.PIE = None
        self.LAMBDA = None
        self.BETA = None

    def low(self) -> NoReturn:
        """
        Compute low precision values for the moon location method calculations.
        """
        jd = np.asarray(self.time.jd)
        T = (jd - _Constants.J2000) / 36525.  # jul cent. since J2000.0

        lambd = (218.32 + 481267.883 * T
                 + 6.29 * np.sin(np.deg2rad(134.9 + 477198.85 * T))
                 - 1.27 * np.sin(np.deg2rad(259.2 - 413335.38 * T))
                 + 0.66 * np.sin(np.deg2rad(235.7 + 890534.23 * T))
                 + 0.21 * np.sin(np.deg2rad(269.9 + 954397.70 * T))
                 - 0.19 * np.sin(np.deg2rad(357.5 + 35999.05 * T))
                 - 0.11 * np.sin(np.deg2rad(186.6 + 966404.05 * T)))
        self.LAMBDA = np.deg2rad(lambd)

        beta = (5.13 * np.sin(np.deg2rad(93.3 + 483202.03 * T))
                + 0.28 * np.sin(np.deg2rad(228.2 + 960400.87 * T))
                - 0.28 * np.sin(np.deg2rad(318.3 + 6003.18 * T))
                - 0.17 * np.sin(np.deg2rad(217.6 - 407332.20 * T)))
        self.BETA = np.deg2rad(beta)

        pie = (0.9508 + 0.0518 * np.cos(np.deg2rad(134.9 + 477198.85 * T))
               + 0.0095 * np.cos(np.deg2rad(259.2 - 413335.38 * T))
               + 0.0078 * np.cos(np.deg2rad(235.7 + 890534.23 * T))
               + 0.0028 * np.cos(np.deg2rad(269.9 + 954397.70 * T)))
        self.PIE = np.deg2rad(pie)

    def high(self) -> NoReturn:
        """
        Compute accurate precession values for the moon location method calculations.
        """
        T = (self.time - 2415020.) / 36525.  # this based around 1900 ... */
        TSQ = T * T
        TCB = TSQ * T
        LPR = 270.434164 + 481267.8831 * T - 0.001133 * TSQ + 0.0000019 * TCB
        M = 358.475833 + 35999.0498 * T - 0.000150 * TSQ - 0.0000033 * TCB
        MPR = 296.104608 + 477198.8491 * T + 0.009192 * TSQ + 0.0000144 * TCB
        D = 350.737486 + 445267.1142 * T - 0.001436 * TSQ + 0.0000019 * TCB
        F = 11.250889 + 483202.0251 * T - 0.003211 * TSQ - 0.0000003 * TCB
        OM = 259.183275 - 1934.1420 * T + 0.002078 * TSQ + 0.0000022 * TCB

        LPR = LPR % 360.
        MPR = MPR % 360.
        M = M % 360.
        D = D % 360.
        F = F % 360.
        OM =OM % 360.

        sinx = np.sin(np.deg2rad(51.2 + 20.2 * T))
        LPR = LPR + 0.000233 * sinx
        M = M - 0.001778 * sinx
        MPR = MPR + 0.000817 * sinx
        D = D + 0.002011 * sinx

        sinx = 0.003964 * np.sin(np.deg2rad(346.560 + 132.870 * T - 0.0091731 * TSQ))
        LPR = LPR + sinx
        MPR = MPR + sinx
        D = D + sinx
        F = F + sinx

        sinx = np.sin(np.deg2rad(OM))
        LPR = LPR + 0.001964 * sinx
        MPR = MPR + 0.002541 * sinx
        D = D + 0.001964 * sinx
        F = F - 0.024691 * sinx
        F = F - 0.004328 * np.sin(np.deg2rad(OM + 275.05 - 2.30 * T))

        e = 1 - 0.002495 * T - 0.00000752 * TSQ

        lambd = (LPR + 6.288750 * np.sin(MPR)
                 + 1.274018 * np.sin(2 * D - MPR)
                 + 0.658309 * np.sin(2 * D)
                 + 0.213616 * np.sin(2 * MPR)
                 - e * 0.185596 * np.sin(M)
                 - 0.114336 * np.sin(2 * F)
                 + 0.058793 * np.sin(2 * D - 2 * MPR)
                 + e * 0.057212 * np.sin(2 * D - M - MPR)
                 + 0.053320 * np.sin(2 * D + MPR)
                 + e * 0.045874 * np.sin(2 * D - M)
                 + e * 0.041024 * np.sin(MPR - M)
                 - 0.034718 * np.sin(D)
                 - e * 0.030465 * np.sin(M + MPR)
                 + 0.015326 * np.sin(2 * D - 2 * F)
                 - 0.012528 * np.sin(2 * F + MPR)
                 - 0.010980 * np.sin(2 * F - MPR)
                 + 0.010674 * np.sin(4 * D - MPR)
                 + 0.010034 * np.sin(3 * MPR)
                 + 0.008548 * np.sin(4 * D - 2 * MPR)
                 - e * 0.007910 * np.sin(M - MPR + 2 * D)
                 - e * 0.006783 * np.sin(2 * D + M)
                 + 0.005162 * np.sin(MPR - D))

        #       /* And furthermore.....*/
        lambd = (lambd + e * 0.005000 * np.sin(M + D)
                 + e * 0.004049 * np.sin(MPR - M + 2 * D)
                 + 0.003996 * np.sin(2 * MPR + 2 * D)
                 + 0.003862 * np.sin(4 * D)
                 + 0.003665 * np.sin(2 * D - 3 * MPR)
                 + e * 0.002695 * np.sin(2 * MPR - M)
                 + 0.002602 * np.sin(MPR - 2 * F - 2 * D)
                 + e * 0.002396 * np.sin(2 * D - M - 2 * MPR)
                 - 0.002349 * np.sin(MPR + D)
                 + e * e * 0.002249 * np.sin(2 * D - 2 * M)
                 - e * 0.002125 * np.sin(2 * MPR + M)
                 - e * e * 0.002079 * np.sin(2 * M)
                 + e * e * 0.002059 * np.sin(2 * D - MPR - 2 * M)
                 - 0.001773 * np.sin(MPR + 2 * D - 2 * F)
                 - 0.001595 * np.sin(2 * F + 2 * D)
                 + e * 0.001220 * np.sin(4 * D - M - MPR)
                 - 0.001110 * np.sin(2 * MPR + 2 * F)
                 + 0.000892 * np.sin(MPR - 3 * D)
                 - e * 0.000811 * np.sin(M + MPR + 2 * D)
                 + e * 0.000761 * np.sin(4 * D - M - 2 * MPR)
                 + e * e * 0.000717 * np.sin(MPR - 2 * M)
                 + e * e * 0.000704 * np.sin(MPR - 2 * M - 2 * D)
                 + e * 0.000693 * np.sin(M - 2 * MPR + 2 * D)
                 + e * 0.000598 * np.sin(2 * D - M - 2 * F)
                 + 0.000550 * np.sin(MPR + 4 * D)
                 + 0.000538 * np.sin(4 * MPR)
                 + e * 0.000521 * np.sin(4 * D - M)
                 + 0.000486 * np.sin(2 * MPR - D))

        B = (5.128189 * np.sin(F)
             + 0.280606 * np.sin(MPR + F)
             + 0.277693 * np.sin(MPR - F)
             + 0.173238 * np.sin(2 * D - F)
             + 0.055413 * np.sin(2 * D + F - MPR)
             + 0.046272 * np.sin(2 * D - F - MPR)
             + 0.032573 * np.sin(2 * D + F)
             + 0.017198 * np.sin(2 * MPR + F)
             + 0.009267 * np.sin(2 * D + MPR - F)
             + 0.008823 * np.sin(2 * MPR - F)
             + e * 0.008247 * np.sin(2 * D - M - F)
             + 0.004323 * np.sin(2 * D - F - 2 * MPR)
             + 0.004200 * np.sin(2 * D + F + MPR)
             + e * 0.003372 * np.sin(F - M - 2 * D)
             + 0.002472 * np.sin(2 * D + F - M - MPR)
             + e * 0.002222 * np.sin(2 * D + F - M)
             + e * 0.002072 * np.sin(2 * D - F - M - MPR)
             + e * 0.001877 * np.sin(F - M + MPR)
             + 0.001828 * np.sin(4 * D - F - MPR)
             - e * 0.001803 * np.sin(F + M)
             - 0.001750 * np.sin(3 * F)
             + e * 0.001570 * np.sin(MPR - M - F)
             - 0.001487 * np.sin(F + D)
             - e * 0.001481 * np.sin(F + M + MPR)
             + e * 0.001417 * np.sin(F - M - MPR)
             + e * 0.001350 * np.sin(F - M)
             + 0.001330 * np.sin(F - D)
             + 0.001106 * np.sin(F + 3 * MPR)
             + 0.001020 * np.sin(4 * D - F)
             + 0.000833 * np.sin(F + 4 * D - MPR))
        #     /* not only that, but */
        B = (B + 0.000781 * np.sin(MPR - 3 * F)
             + 0.000670 * np.sin(F + 4 * D - 2 * MPR)
             + 0.000606 * np.sin(2 * D - 3 * F)
             + 0.000597 * np.sin(2 * D + 2 * MPR - F)
             + e * 0.000492 * np.sin(2 * D + MPR - M - F)
             + 0.000450 * np.sin(2 * MPR - F - 2 * D)
             + 0.000439 * np.sin(3 * MPR - F)
             + 0.000423 * np.sin(F + 2 * D + 2 * MPR)
             + 0.000422 * np.sin(2 * D - F - 3 * MPR)
             - e * 0.000367 * np.sin(M + F + 2 * D - MPR)
             - e * 0.000353 * np.sin(M + F + 2 * D)
             + 0.000331 * np.sin(F + 4 * D)
             + e * 0.000317 * np.sin(2 * D + F - M + MPR)
             + e * e * 0.000306 * np.sin(2 * D - 2 * M - F)
             - 0.000283 * np.sin(MPR + 3 * F))

        OM1 = 0.0004664 * np.cos(np.deg2rad(OM))
        OM2 = 0.0000754 * np.cos(np.deg2rad(OM + 275.05 - 2.30 * T))

        beta = B * (1. - OM1 - OM2)

        self.PIE = (0.950724 + 0.051818 * np.cos(MPR)
                    + 0.009531 * np.cos(2 * D - MPR)
                    + 0.007843 * np.cos(2 * D)
                    + 0.002824 * np.cos(2 * MPR)
                    + 0.000857 * np.cos(2 * D + MPR)
                    + e * 0.000533 * np.cos(2 * D - M)
                    + e * 0.000401 * np.cos(2 * D - M - MPR)
                    + e * 0.000320 * np.cos(MPR - M)
                    - 0.000271 * np.cos(D)
                    - e * 0.000264 * np.cos(M + MPR)
                    - 0.000198 * np.cos(2 * F - MPR)
                    + 0.000173 * np.cos(3 * MPR)
                    + 0.000167 * np.cos(4 * D - MPR)
                    - e * 0.000111 * np.cos(M)
                    + 0.000103 * np.cos(4 * D - 2 * MPR)
                    - 0.000084 * np.cos(2 * MPR - 2 * D)
                    - e * 0.000083 * np.cos(2 * D + M)
                    + 0.000079 * np.cos(2 * D + 2 * MPR)
                    + 0.000072 * np.cos(4 * D)
                    + e * 0.000064 * np.cos(2 * D - M + MPR)
                    - e * 0.000063 * np.cos(2 * D + M - MPR)
                    + e * 0.000041 * np.cos(M + D)
                    + e * 0.000035 * np.cos(2 * MPR - M)
                    - 0.000033 * np.cos(3 * MPR - 2 * D)
                    - 0.000030 * np.cos(MPR + D)
                    - 0.000029 * np.cos(2 * F - 2 * D)
                    - e * 0.000029 * np.cos(2 * MPR + M)
                    + e * e * 0.000026 * np.cos(2 * D - 2 * M)
                    - 0.000023 * np.cos(2 * F - 2 * D + MPR)
                    + e * 0.000019 * np.cos(4 * D - M - MPR))

        self.BETA = Angle(np.deg2rad(beta), unit=u.rad)
        self.LAMBDA = Angle(np.deg2rad(lambd), unit=u.rad)

#TODO: static method in class above if is not using elsewhere
def _current_geocent_frame(time: Time)-> :
    """ 
    Returns a PrecessedGeocentric frame for the equinox
    spcified by the time.

    Parameters
    ----------

    time : astropy Time, if an array then the first entry is used

    Returns

    an astropy PrecessedGeocentric time.
    """
    # Generate a PrecessedGeocentric frame for the current equinox.
    time_ep = 2000. + (np.asarray(time.jd) - _Constants.J2000) / 365.25
    if time_ep.ndim == 0:
        time_ep = time_ep[None]  # Makes 1D
    #eq = Time("J{:7.2f}".format(time_ep[0]))
    equinox = Time(f'J{time_ep[0]:7.2f}')
    # print(eq)
    return PrecessedGeocentric(equinox=equinox)


def _geocentric_coors(geolong: Angle, geolat: float, height: float) -> Tuple[float, float, float]:
    """
    
    geocentric XYZ coordinates for a location at a longitude,
    latitude and height above sea level.

    Retained because if one replaces the longitude input with the
    sidereal time, the return is the XYZ in the equatorial frame
    of date.  This is used in the lunar topocentric correction.

    Parameters
    ----------

    geolong : Angle
        Geographic longitude, or LST to get celestial-aligned result
    geolat :  Angle
        Geographic latitude
    height :  float
        Height above sea level, which must be in meters.

    Returns: Tuple of astropy Quantities X, Y, Z, which
        are distances.
    """

    # computes the geocentric coordinates from the geodetic
    # (standard map-type) longitude, latitude, and height.
    # Notation generally follows 1992 Astr Almanac, p. K11 */
    # NOTE that if you replace "geolong" with the local sidereal
    # time, this automatically gives you XYZ in the equatorial frame
    # of date.
    # In this version, geolong and geolat are assumed to be
    # Angles and height is assumed to be in meters; returns
    # a triplet of explicit Distances.

    denom = (1. - _Constants.FLATTEN) * np.sin(geolat)
    denom = np.cos(geolat) * np.cos(geolat) + denom * denom
    C_geo = 1. / np.sqrt(denom)
    S_geo = (1. - _Constants.FLATTEN) * (1. - _Constants.FLATTEN) * C_geo
    C_geo = C_geo + height / _Constants.EQUAT_RAD
    #  deviation from almanac notation -- include height here.
    S_geo = S_geo + height / _Constants.EQUAT_RAD
    # distancemultiplier = Distance(_Constants.EQUAT_RAD, unit = u.m)
    x_geo = _Constants.EQUAT_RAD * C_geo * np.cos(geolat) * np.cos(geolong)
    y_geo = _Constants.EQUAT_RAD * C_geo * np.cos(geolat) * np.sin(geolong)
    z_geo = _Constants.EQUAT_RAD * S_geo * np.sin(geolat)

    return x_geo, y_geo, z_geo

def scalar_input(func):
    """
    Decorator to convert a function that returns a tuple to a function that returns a scalar.
    """

    def wrapper(*args, **kwargs):

        #transform the input to numpy
        np_args = [np.asarray(arg) for arg in args]
        if any( arg.ndim == 0 for arg in np_args):
            args = [arg[np.newaxis] for arg in np_args]
            return np.squeeze(func(*args, **kwargs))
        return func(*args, **kwargs)
    return wrapper


def min_max_alt(lat, dec) -> Tuple[Angle, Angle]:
    """Finds the mimimum and maximum altitudes of a celestial location.

    Parameters
    ----------
    lat : astropy Angle
        Latitude of site.
    dec : astropy Angle, float or array
        declination of the object.

    Returns :

    (minalt, maxalt) : both Astropy Angle
        tuple of minimum and maximum altitudes.
    """

    # arguments are Angles; returns (minalt, maxalt) as Angles
    # where min and max are the minimum and maximum altitude
    # an object at declination dec reaches at this latitude.

    dec = np.asarray(dec.to_value(u.rad).data)
    scalar_input = False
    if dec.ndim == 0:
        dec = dec[None]
        scalar_input = True

    maxalt = np.zeros(len(dec))
    minalt = np.zeros(len(dec))

    x = np.cos(dec) * np.cos(lat) + np.sin(dec) * np.sin(lat)
    ii = np.where(abs(x) <= 1.)[0][:]
    if len(ii) != 0:
        maxalt[ii] = np.arcsin(x[ii])

    x = np.sin(dec) * np.sin(lat) - np.cos(dec) * np.cos(lat)
    ii = np.where(abs(x) <= 1.)[0][:]
    if len(ii) != 0:
        minalt[ii] = np.arcsin(x[ii])

    if scalar_input:
        minalt = np.squeeze(minalt)
        maxalt = np.squeeze(maxalt)
    return Angle(minalt, unit=u.rad), Angle(maxalt, unit=u.rad)


def local_midnight_time(aTime, localtzone):
    """find nearest local midnight (UT).

    If it's before noon local time, returns previous midnight;
    if after noon, return next midnight.

    Parameters :

    aTime : astropy Time

    localtzone : timezone object.

    Returns

    Time.  This is not zone-aware, but should be correct.
    """

    # takes an astropy Time and the local time zone and
    # generates another Time which is the nearest local
    # clock-time midnight.  The returned Time is unaware
    # of the timezone but should be correct.

    aTime = Time(np.asarray(aTime.iso), format='iso')
    scalar_input = False
    if aTime.ndim == 0:
        aTime = aTime[None]  # Makes 1D
        scalar_input = True

    datetmid = []
    for time in aTime:
        datet = time.to_datetime(timezone=localtzone)
        # print(datet)

        # if before midnight, add 12 hours
        if datet.hour >= 12:
            datet = datet + timedelta(hours=12.)

        datetmid.append(localtzone.localize(datetime(datet.year, datet.month, datet.day, 0, 0, 0)))

    if scalar_input:
        result = Time(datetmid[0])
    else:
        result = Time(datetmid)
    return result

 
def local_sidereal_time(time: Time, location: EarthLocation) -> Angle:
    """
    moderate-precision (1 sec) local sidereal time

    Adapted with minimal changes from skycalc routine. Native
    astropy routines are unnecessarily precise for our purposes and
    rather slow.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time at which to compute the local sidereal time.
    location : `~astropy.coordinates.EarthLocation`
        Location on Earth for which to compute the local sidereal time.

    Returns
    -------
    lst : `~astropy.coordinates.Angle`
        Local sidereal time.
    """
    # julian date represented as integer values
    julian_int = np.asarray(time.jd, dtype=int)

    scalar_input = False
    # check if time is an array or a scalar 
    if julian_int.ndim == 0:
        julian_int = julian_int[None]
        scalar_input = True
    
    fraction = time.jd - julian_int
    mid = fraction + 0.5
    ut = fraction - 0.5
    less_than_half = np.where(fraction < 0.5)[0][:]
    if len(less_than_half) != 0:
        mid[less_than_half] = julian_int[less_than_half] - 0.5
        ut[less_than_half] = fraction[less_than_half] + 0.5  # as fraction of a day.
    
    t = (mid - _Constants.J2000) / 36525.

    sidereal = (24110.54841 + 8640184.812866 * t + 0.093104 * t ** 2 - 6.2e-6 * t ** 3) / 86400.
    # at Greenwich
    sid_int = sidereal.astype(np.int)
    sidereal -= sid_int
    # longitude is measured east so add.
    sidereal += (1.0027379093 * ut + location.lon.hour / 24.)
    
    sid_int = sidereal.astype(np.int)
    sidereal = (sidereal - sid_int) * 24.
    # if(sid < 0.) : sid = sid + 24.
    # TODO: A convertion to radians is needed if the output is not an Angle
    lst = Angle(sidereal, unit=u.hour)
    lst.wrap_at(24. * u.hour, inplace=True)

    if scalar_input:
        return np.squeeze(lst)
    return lst


def moon_location(time: Time, obs: EarthLocation, precision='Accurate') -> Tuple[SkyCoord, float]:  
    """  
    Compute topocentric location and distance of moon to better accuracy.

    This is good to about 0.01 degrees

    Parameters
    ----------

    time : Time
        An astropy Time.  This is converted to TT (terrestrial time) internally
        for the computations.
    obs : EarthLocation
        location on earth.

    Returns:

    tuple of a SkyCoord and a distance.

    """

    #  More accurate (but more elaborate and slower) lunar
    #   ephemeris, from Jean Meeus' *Astronomical Formulae For Calculators*,
    #   pub. Willman-Bell.  Includes all the terms given there. */

    # a run of comparisons every 3 days through 2018 shows that this
    # agrees with the more accurate astropy positions with an RMS of
    # 0.007 degrees.

    # Terrestrial time with julian day
    time_ttjd = np.asarray(time.tt.jd)  # important to use correct time argument for this!!
    scalar_input = False
    if time_ttjd.ndim == 0:
        time_ttjd = time_ttjd[None]  # Makes 1D
        scalar_input = True

    def high_precision_moon_location():
        dist = Distance(1. / np.sin(np.deg2rad(moon_vars.PIE)) * _Constants.EQUAT_RAD)

        # place these in a skycoord in ecliptic coords of date.  Handle distance
        # separately since it does not transform properly for some reason.

        # eq = 'J{:7.2f}'.format(2000. + (time_ttjd[0] - _Constants.J2000) / 365.25)
        equinox = f'J{2000. + (time_ttjd[0] - _Constants.J2000) / 365.25:7.2f}'
        frame = GeocentricTrueEcliptic(equinox=equinox)
        inecl = SkyCoord(lon=Angle(moon_vars.LAMBDA, unit=u.rad), lat=Angle(moon_vars.BETA, unit=u.rad), frame=frame)

        # Transform into geocentric equatorial.
        geocen = inecl.transform_to(_current_geocent_frame(time))

        # Do the topo correction yourself. First form xyz coords in equatorial syst of date
        x = dist * np.cos(geocen.ra) * np.cos(geocen.dec)
        y = dist * np.sin(geocen.ra) * np.cos(geocen.dec)
        z = dist * np.sin(geocen.dec)

        # Now compute geocentric location of the observatory in a frame aligned with the
        # equatorial system of date, which one can do simply by replacing the west longitude
        # with the sidereal time

        # Exact match with thorskyutil/skycalc with the line below
        xobs, yobs, zobs = _geocentric_coors(local_sidereal_time(time, obs), obs.lat, obs.height)

        # recenter moon's cartesian coordinates on position of obs
        x = x - xobs
        y = y - yobs
        z = z - zobs

        # form the toposcentric ra and dec and bundle them into a skycoord of epoch of date.
        topo_dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        raout = np.arctan2(y, x)
        decout = np.arcsin(z / topo_dist)

        if scalar_input:
            raout = np.squeeze(raout)
            decout = np.squeeze(decout)
            topodist = np.squeeze(topo_dist)
        return SkyCoord(raout, decout, unit=u.rad, frame=_current_geocent_frame(time)), topo_dist
    

    def low_precision_moon_location():
        # This is the same as the high precision method, but with a
        # different set of coefficients.  The difference is small.
        # Good to about 0.1 deg, from the 1992 Astronomical Almanac, p. D46.
        # Note that input time is a float.


        # Terrestrial time with julian day
        sid = local_sidereal_time(time, obs)
        lat = obs.lat
        distance = 1. / np.sin(moon_vars.PIE)

        l = np.cos(moon_vars.BETA) * np.cos(moon_vars.LAMBDA)
        m = 0.9175 * np.cos(moon_vars.BETA) * np.sin(moon_vars.LAMBDA) - 0.3978 * np.sin(moon_vars.BETA)
        n = 0.3978 * np.cos(moon_vars.BETA) * np.sin(moon_vars.LAMBDA) + 0.9175 * np.sin(moon_vars.BETA)

        x = l * distance
        y = m * distance
        z = n * distance  # /* for topocentric correction */
  
        x = x - np.cos(lat) * np.cos(sid)
        y = y - np.cos(lat) * np.sin(sid)
        z = z - np.sin(lat)

        topo_dist = np.sqrt(x * x + y * y + z * z)

        l = x / topo_dist
        m = y / topo_dist
        n = z / topo_dist

        alpha = np.arctan2(m, l)
        delta = np.arcsin(n)
        distancemultiplier = Distance(_Constants.EQUAT_RAD, unit=u.m)

        fr = _current_geocent_frame(time)
        return SkyCoord(alpha, delta, topo_dist * distancemultiplier, frame=fr), topo_dist


    if precision == 'Accurate':
        moon_vars =  _MoonLocationVariables(time_ttjd).high()
        return high_precision_moon_location(time_ttjd)
    elif precision == 'Low':
        moon_vars =  _MoonLocationVariables(time_ttjd).low()
        return low_precision_moon_location()
    else:
        raise ValueError(f'Precision must be either "Accurate" or "Low".  Got {precision}')



def sun_location(time: Time) -> SkyCoord:
    """low-precision position of the sun.

    Good to about 0.01 degree, from the 1990 Astronomical Almanac p. C24.
    At this level topocentric correction is not needed.

    Paramters
    ---------
    time : astropy Time

    Returns

    a SkyCoord in the geocentric frame of epoch of date.

    """

    # Low precision formulae for the sun, from Almanac p. C24 (1990) */
    # said to be good to about a hundredth of a degree.

    jd = np.asarray(time.jd)
    scalar_input = False
    if jd.ndim == 0:
        jd = jd[None]
        scalar_input = True

    n = jd - _Constants.J2000  # referred to J2000
    L = 280.460 + 0.9856474 * n
    g = np.deg2rad(357.528 + 0.9856003 * n)
    lambd = np.deg2rad(L + 1.915 * np.sin(g) + 0.020 * np.sin(2. * g))
    epsilon = np.deg2rad(23.439 - 0.0000004 * n)

    x = np.cos(lambd)
    y = np.cos(epsilon) * np.sin(lambd)
    z = np.sin(epsilon) * np.sin(lambd)

    ra = np.arctan2(y, x)
    dec = np.arcsin(z)

    fr = _current_geocent_frame(time)


    if scalar_input:
        ra = np.squeeze(ra)
        dec = np.squeeze(dec)
    return SkyCoord(ra, dec, frame=fr, unit='radian')


def altitude_above(dec: Union[Angle,float, npt.array], 
                ha: Union[Angle,float, npt.array], 
                lat: Union[Angle,float, npt.array]) -> Tuple[Angle, Angle, Angle]:
    """
    Compute altitude above horizon, azimuth, and parallactic angle.

    The parallactic angle is the position angle of the arc between the
    object and the zenith, i.e. the position angle that points 'straight up'
    when you're looking at an object.  It is important for atmospheric
    refraction and dispersion compensation, which Filippenko discusses
    ( 1982PASP...94..715F ).  Does not take small effects into
    account (polar motion, nutation, whatever) so is Lighter weight than
    the astropy equivalents.

    Filippenko's expression for the parallactic angle leaves it to the the user
    to select the correct root of an inverse trig function; this is handled
    automatically here by fully solving the astronomical triangle.

    Parameters
    ----------

    dec : Angle, float or numpy array
       Declination
    ha : Angle, float or numpy array
       Hour angle (spherical astronomy) of the position, positive westward
    lat : Angle
    Latitude of site.

    Returns

    tuple of (altitude, azimuth, parallactic), all of which are Angles.

    """
    # The astropy altaz transformation depends on the 3 Mbyte
    # download from USNO to find the lst, so here is a stripped
    # down version.
    # Arguments are all assumed to be Angles so they don't need
    # to be converted to radians; the dec is assumed to be in
    # equinox of date to avoid
    # We get the parallactic angle almost for since we have
    # ha, colat, and altitude.

    dec = np.asarray(dec.to_value(u.rad).data) * u.rad
    ha = np.asarray(ha.to_value(u.rad)) * u.rad
    scalar_input = False
    if dec.ndim == 0 and ha.ndim == 0:
        scalar_input = True
    if dec.ndim == 0:
        dec = dec[None]
    if ha.ndim == 0:
        ha = ha[None]

    if len(dec) == 1 and len(ha) > 1:
        dec = dec * np.ones(len(ha))
    elif len(dec) > 1 and len(ha) == 1:
        ha = ha * np.ones(len(dec))
    elif len(dec) != len(ha):
        print('Error: dec and ha have incompatible lengths')
        return

    cosdec = np.cos(dec)
    sindec = np.sin(dec)
    cosha = np.cos(ha)
    sinha = np.sin(ha)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    altit = Angle(np.arcsin(cosdec * cosha * coslat + sindec * sinlat), unit=u.radian)

    y = sindec * coslat - cosdec * cosha * sinlat  # due north component
    z = -1. * cosdec * sinha  # due east component
    az = Longitude(np.arctan2(z, y), unit=u.radian)

    # now solve the spherical trig to give parallactic angle

    parang = np.zeros(len(dec))
    ii = np.where(cosdec != 0.0)[0][:]
    if len(ii) != 0:
        sinp = -1. * np.sin(az[ii]) * coslat / cosdec[ii]
        # spherical law of sines .. note cosdec = sin of codec,
        # coslat = sin of colat ....
        cosp = -1. * np.cos(az[ii]) * cosha[ii] - np.sin(az[ii]) * sinha[ii] * sinlat
        # spherical law of cosines ... also expressed in
        # already computed variables.
        parang[ii] = np.arctan2(sinp, cosp)
    jj = np.where(cosdec == 0.0)[0][:]  # you're on the pole
    if len(jj) != 0:
        parang[jj] = np.pi

    if scalar_input:
        altit = np.squeeze(altit)
        az = np.squeeze(az)
        parang = np.squeeze(parang)
    return altit, az, Angle(parang, unit=u.rad)


def true_airmass(altit: Angle) -> np.ndarray:
    """true airmass for an altitude.
    Equivalent of getAirmass in the QPT, based on vskyutil.true_airmass
    https://github.com/gemini-hlsw/ocs/blob/12a0999bc8bb598220ddbccbdbab5aa1e601ebdd/bundle/edu.gemini.qpt.client/src/main/java/edu/gemini/qpt/core/util/ImprovedSkyCalcMethods.java#L119

    Based on a fit to Kitt Peak airmass tables, C. M. Snell & A. M. Heiser, 1968,
    PASP, 80, 336.  Valid to about airmass 12, and beyond that just returns
    secz minus 1.5, which won't be quite right.

    Parameters
    ----------

    altit : Angle, float or numpy array
        Altitude above horizon.

    Returns : float
    """
    # takes an Angle and return the true airmass, based on
    # 	 a tabulation of the mean KPNO
    #            atmosphere given by C. M. Snell & A. M. Heiser, 1968,
    # 	   PASP, 80, 336.  They tabulated the airmass at 5 degr
    #            intervals from z = 60 to 85 degrees; I fit the data with
    #            a fourth order poly for (secz - airmass) as a function of
    #            (secz - 1) using the IRAF curfit routine, then adjusted the
    #            zeroth order term to force (secz - airmass) to zero at
    #            z = 0.  The poly fit is very close to the tabulated points
    # 	   (largest difference is 3.2e-4) and appears smooth.
    #            This 85-degree point is at secz = 11.47, so for secz > 12
    #            just return secz   */

    #    coefs = [2.879465E-3,  3.033104E-3, 1.351167E-3, -4.716679E-5]

    altit = np.asarray(altit.to_value(u.rad).data)
    scalar_input = False
    if altit.ndim == 0:
        altit = altit[None]  # Makes 1D
        scalar_input = True

    # ret = np.zeros(len(altit))
    ret = np.full(len(altit), 58.)
    ii = np.where(altit > 0.0)[0][:]
    if len(ii) != 0:
        ret[ii] = 1. / np.sin(altit[ii])  # sec z = 1/sin (altit)

    kk = np.where(np.logical_and(ret >= 0.0, ret < 12.))[0][:]
    if len(kk) != 0:
        seczmin1 = ret[kk] - 1.
        coefs = np.array([-4.716679E-5, 1.351167E-3, 3.033104E-3, 2.879465E-3, 0.])
        ret[kk] = ret[kk] - np.polyval(coefs, seczmin1)
        # print "poly gives",  np.polyval(coefs,seczmin1)

    if scalar_input:
        return np.squeeze(ret)
    return ret


def hour_angle(dec, lat, alt):
    """
    Return an Angle giving the hour angle (from spherical astronomy, east
    or west of meridian, not the u.hourangle from astropy) at which
    the declination dec reaches altitude alt.

    If the object is always above alt, an Angle of +1000 radians is returned.
    If always below, -1000 radians.

    Parameters :
    dec : Angle, float or array
       Declination of source.
    lat : Angle
       Latitude of site.
    alt : Angle, float or array
       Height above horizon for computation.

    dec and alt must have the same dimensions
    """

    # Arguments are all angles.
    # returns hour angle at which object at dec is at altitude alt for a
    # latitude lat.

    dec = np.asarray(dec.to_value(u.rad).data) * u.rad
    alt = np.asarray(alt.to_value(u.rad)) * u.rad
    # dec = np.asarray(dec.rad) * u.rad
    # alt = np.asarray(alt.rad) * u.rad
    scalar_input = False
    if dec.ndim == 0 and alt.ndim == 0:
        scalar_input = True
    if dec.ndim == 0:
        dec = dec[None]
    if alt.ndim == 0:
        alt = alt[None]

    if len(dec) == 1 and len(alt) > 1:
        dec = dec * np.ones(len(alt))
    elif len(dec) > 1 and len(alt) == 1:
        alt = alt * np.ones(len(dec))
    elif len(dec) != len(alt):
        print('Error: dec and alt have incompatible lengths')
        return

    x = np.zeros(len(dec))
    codec = np.zeros(len(dec))
    zdist = np.zeros(len(dec))

    minalt, maxalt = min_max_alt(lat, dec)
    ii = np.where(alt < minalt)[0][:]
    if len(ii) != 0:
        x[ii] = -1000.

    jj = np.where(alt > maxalt)[0][:]
    if len(jj) != 0:
        x[jj] = 1000.

    kk = np.where(np.logical_and(alt >= minalt, alt <= maxalt))[0][:]
    if len(kk) != 0:
        rightang = Angle(np.pi / 2, unit=u.rad)
        codec[kk] = rightang - dec[kk]
        colat = rightang - lat
        zdist[kk] = rightang - alt[kk]
        x[kk] = (np.cos(zdist[kk]) - np.cos(codec[kk]) * np.cos(colat)) / (np.sin(codec[kk]) * np.sin(colat))
        x[kk] = np.arccos(x[kk])

    if scalar_input:
        # return (Angle(np.squeeze(x), unit = u.rad))
        x = np.squeeze(x)
    return Angle(x, unit=u.rad)


def sun_crosses(alt, tguess, location) -> Optional[Time]:
    """
    time at which the sun crosses a given elevation.

    Parameters:

    alt : Angle, single or array. If array, then must be the same length as tguess
       Desired altitude.
    tguess : Time, single or array
       Starting time for iteration.  This must be fairly
       close so that the iteration coverges on the correct
       phenomenon (e.g., rise time, not set time).
    location : EarthLocation

    Returns: Time if convergent
            None if non-convergent
    """

    # returns the Time at which the sun crosses a
    # particular altitude alt, which is an Angle,
    # for an EarthLocation location.

    # This of course happens twice a day (or not at
    # all); tguess is a Time approximating the answer.
    # The usual use case will be to compute roughly when
    # sunset or twilight occurs, and hand the result to this
    # routine to get a more exact answer.

    # This uses the low-precision sun "lpsun", which is
    # typically good to 0.01 degree.  That's plenty good
    # enough for computing rise, set, and twilight times.

    tguess = Time(np.asarray(tguess.jd), format='jd')
    alt = Angle(np.asarray(alt.to_value(u.rad)), unit=u.rad)
    scalar_input = False
    if tguess.ndim == 0 and alt.ndim == 0:
        scalar_input = True
    if tguess.ndim == 0:
        tguess = tguess[None]  # Makes 1D
    if alt.ndim == 0:
        alt = alt[None]

    if len(tguess) == 1 and len(alt) > 1:
        tguess = Time(tguess.jd * np.ones(len(alt)), format='jd')
    elif len(tguess) > 1 and len(alt) == 1:
        alt = alt * np.ones(len(tguess))
    elif len(tguess) != len(alt):
        print('Error: alt and tguess have incompatible lengths')
        return

    sunpos = sun_location(tguess)
    # print "sunpos entering",sunpos
    # print "tguess.jd, longit:",tguess.jd, location.lon.hour
    tolerance = Angle(1.0e-4, unit=u.rad)

    delt = TimeDelta(0.002, format='jd')  # timestep
    # print "sidereal: ",local_sidereal_time(tguess, location)
    # print "sunpos.ra: ",sunpos.ra

    ha = local_sidereal_time(tguess, location) - sunpos.ra
    # print "ha entering",ha
    alt2, az, parang = altitude_above(sunpos.dec, Angle(ha, unit=u.hourangle), location.lat)
    # print "alt2",alt2
    tguess = tguess + delt
    sunpos = local_sidereal_time(tguess)
    # print "sunpos with delt",sunpos
    alt3, az, parang = altitude_above(sunpos.dec, local_sidereal_time(tguess, location) - sunpos.ra, location.lat)
    err = alt3 - alt
    # print "alt3, alt, err",alt3,alt,err
    deriv = (alt3 - alt2) / delt
    # print "deriv",deriv
    kount = np.zeros(len(tguess), dtype=int)
    kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]
    while (len(kk) != 0):
        tguess[kk] = tguess[kk] - err[kk] / deriv[kk]
        sunpos = None
        sunpos = sun_location(tguess[kk])
        alt3[kk], az[kk], parang[kk] = altitude_above(sunpos.dec, local_sidereal_time(tguess[kk], location) - sunpos.ra,
                                                   location.lat)
        err[kk] = alt3[kk] - alt[kk]
        kount[kk] = kount[kk] + 1
        ii = np.where(kount >= 9)[0][:]
        if len(ii) != 0:
            print("Sunrise, set, or twilight calculation not converging!\n")
            return None
        kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]

    if scalar_input:
        tguess = np.squeeze(tguess)
    return Time(tguess, format='iso')


def jd_moon_alt(alt, tguess, location):
    """
    
    Time at which moon passes a given altitude.

    This really does have to be iterated since the moon moves fairly
    quickly.

    Parameters
    ----------
    alt : Angle, single or array.  If array, then must be the same length as tguess
       desired altitude.
    tguess : Time, single or array
       initial guess; this needs to be fairly close.
    location : EarthLocation

    Returns
       a Time, or None if non-convergent.
    """

    # tguess is a Time, location is an EarthLocation

    tguess = Time(np.asarray(tguess.jd), format='jd')
    scalar_input = False
    if tguess.ndim == 0:
        tguess = tguess[None]  # Makes 1D
        scalar_input = True
    alt = Angle(np.asarray(alt.to_value(u.rad)), unit=u.rad)
    if alt.ndim == 0:
        alt = alt[None]

    if len(tguess) != len(alt):
        print('Error: alt and guess must be the same length')
        return

    moonpos, topodist = moon_location(tguess, location)
    # print "npos entering",moonpos
    # print "tguess.jd, longit:",tguess.jd, location.lon.hour
    tolerance = Angle(1.0e-4, unit=u.rad)

    delt = TimeDelta(0.002, format='jd')  # timestep
    # print "sidereal: ",local_sidereal_time(tguess, location)
    # print "moonpos.ra: ",moonpos.ra

    ha = local_sidereal_time(tguess, location) - moonpos.ra
    # print "ha entering",ha
    alt2, az, parang = altitude_above(moonpos.dec, Angle(ha, unit=u.hourangle), location.lat)
    # print "alt2",alt2
    tguess = tguess + delt
    moonpos, topodist = moon_location(tguess, location)
    # print "moonpos with delt",moonpos
    alt3, az, parang = altitude_above(moonpos.dec, local_sidereal_time(tguess, location) - moonpos.ra, location.lat)
    err = alt3 - alt
    # print "alt3, alt, err",alt3,alt,err
    deriv = (alt3 - alt2) / delt
    # print "deriv",deriv
    kount = np.zeros(len(tguess), dtype=int)
    kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]
    while (len(kk) != 0):
        # print "iterating: err = ",err," kount = ",kount
        tguess[kk] = tguess[kk] - err[kk] / deriv[kk]
        moonpos = None
        topodist = None
        moonpos, topodist = moon_location(tguess[kk], location)
        alt3[kk], az[kk], parang[kk] = altitude_above(moonpos.dec, local_sidereal_time(tguess[kk], location) - moonpos.ra,
                                                   location.lat)
        err[kk] = alt3[kk] - alt[kk]
        ii = np.where(kount >= 9)[0][:]
        if len(ii) != 0:
            print("Moonrise or set calculation not converging!\n")
            return None
        kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]

    # print "out, err = ",err
    if scalar_input:
        tguess = np.squeeze(tguess)
    return Time(tguess, format='iso')


def night_events(aTime, location, localtzone, verbose=True):
    """
    Compute phenomena for a given night.

    This is mostly a testbed that prints results directly.

    Parameters
    ----------
    aTime : astropy Time, single or array
        input time; if before noon, events of previous night are computed.
    location : EarthLocation
    localtzone : timezone object.
    verbose: verbose output
    """
    # prototype for the events of a single night -- sunset and rise,
    # twilights, and moonrise and set.

    aTime = Time(np.asarray(aTime.iso), format='iso')
    scalar_input = False
    if aTime.ndim == 0:
        aTime = aTime[None]  # Makes 1D
        scalar_input = True

    nt = len(aTime)

    midnight = local_midnight_time(aTime, localtzone)  # nearest clock-time midnight (UT)
    # print("midnight:", midnight)
    lstmid = local_sidereal_time(midnight, location)

    # Sun
    sunmid = sun_location(midnight)
    sunpos = sun_location(aTime)

    # allow separate rise and set altitudes for horizon effects
    horiz = (-0.883 - np.sqrt(2. * location.height / _Constants.EQUAT_RAD) * (180. / np.pi)) * u.deg
    # print(horiz)
    setalt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    risealt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    twialt12 = Angle(-12. * np.ones(nt), unit=u.deg)  # 12 degree nautical twilight

    # print(sunmid.dec, setalt)
    sunsetha = hour_angle(sunmid.dec, location.lat, setalt)  # corresponding hr angles
    sunriseha = Angle(2. * np.pi, unit=u.rad) - hour_angle(sunmid.dec, location.lat, risealt)  # corresponding hr angles
    twilightha12 = hour_angle(sunmid.dec, location.lat, twialt12)

    hasunmid = (lstmid - sunmid.ra).wrap_at(24. * u.hour)
    # nightcenter = midnight - TimeDelta(hasunmid.hour / 24. - 0.5, format='jd')
    # print "nightcenter", nightcenter

    sunsetguess = hasunmid - sunsetha  # angles away from midnight
    sunriseguess = sunriseha - hasunmid
    evetwiguess12 = hasunmid - twilightha12
    morntwiguess12 = Angle(2. * np.pi, unit=u.rad) - twilightha12 - hasunmid

    # print "setguess: ",setguess
    # print "twiguess: ", twiguess

    # convert to time deltas
    TDsunset = TimeDelta(sunsetguess.hour / 24., format='jd')
    TDsunrise = TimeDelta(sunriseguess.hour / 24., format='jd')
    TDevetwi12 = TimeDelta(evetwiguess12.hour / 24., format='jd')
    TDmorntwi12 = TimeDelta(morntwiguess12.hour / 24., format='jd')
    # form into times and iterate to accurate answer.

    tsunset = midnight - TDsunset  # first approx
    tsunset = sun_crosses(setalt, tsunset, location)

    tsunrise = midnight + TDsunrise  # first approx
    tsunrise = sun_crosses(risealt, tsunrise, location)

    tevetwi12 = midnight - TDevetwi12
    tevetwi12 = sun_crosses(twialt12, tevetwi12, location)

    tmorntwi12 = midnight + TDmorntwi12
    tmorntwi12 = sun_crosses(twialt12, tmorntwi12, location)

    if verbose:
        print("sunset: ", tsunset)
        print("sunrise: ", tsunrise)
        print("eve twi12: ", tevetwi12)
        print("morn twi12:", tmorntwi12)
    # Moon
    moonmid = moon_location(midnight, location, precision='Low')
    hamoonmid = lstmid - moonmid.ra
    hamoonmid.wrap_at(12. * u.hour, inplace=True)
    moonpos, topodist = moon_location(aTime, location)

    if verbose:
        print("moon at midnight: ", moonmid.to_string('hmsdms'))
        print("hamoonmid: ", hamoonmid.hour, 'hr')

    # roughlunarday = TimeDelta(1.0366, format='jd')

    moonsetha = hour_angle(moonmid.dec, location.lat, setalt)  # corresponding hr angles
    moonsetdiff = moonsetha - hamoonmid  # how far from setting point at midn.
    # find nearest setting point
    # if moonsetdiff.hour >= 12. : moonsetdiff = moonsetdiff - Angle(24. * u.hour)
    ii = np.where(moonsetdiff.hour >= 12.)[0][:]
    if len(ii) != 0:
        moonsetdiff[ii] = moonsetdiff[ii] - Angle(24. * u.hour)

    # if moonsetdiff.hour < -12. : moonsetdiff = moonsetdiff + Angle(24. * u.hour)
    jj = np.where(moonsetdiff.hour < -12.)[0][:]
    if len(jj) != 0:
        moonsetdiff[jj] = moonsetdiff[jj] + Angle(24. * u.hour)

    TDmoonset = TimeDelta(moonsetdiff.hour / 24., format='jd')
    tmoonset = midnight + TDmoonset
    if verbose: print("moonset first approx:", tmoonset)
    tmoonset = jd_moon_alt(setalt, tmoonset, location)
    # if verbose: print("moonset: ", tmoonset)
    if verbose: print("moonset: ", tmoonset)

    moonriseha = -1. * hour_angle(moonmid.dec, location.lat, risealt)  # signed
    moonrisediff = moonriseha - hamoonmid  # how far from riseting point at midn.
    # find nearest riseting point
    # if moonrisediff.hour >= 12.: moonrisediff = moonrisediff - Angle(24. * u.hour)
    # if moonrisediff.hour < -12.: moonrisediff = moonrisediff + Angle(24. * u.hour)
    ii = np.where(moonrisediff.hour >= 12.)[0][:]
    if len(ii) != 0:
        moonrisediff[ii] = moonrisediff[ii] - Angle(24. * u.hour)
    jj = np.where(moonrisediff.hour < -12.)[0][:]
    if len(jj) != 0:
        moonrisediff[jj] = moonrisediff[jj] + Angle(24. * u.hour)

    TDmoonrise = TimeDelta(moonrisediff.hour / 24., format='jd')
    tmoonrise = midnight + TDmoonrise
    if verbose: print("moonrise first approx:", tmoonrise)
    tmoonrise = jd_moon_alt(risealt, tmoonrise, location)
    # if verbose: print("moonrise: ", tmoonrise)
    if verbose: print("moonrise: ", tmoonrise)

    if scalar_input:
        tsunset = np.squeeze(tsunset)
        tsunrise = np.squeeze(tsunrise)
        tevetwi12 = np.squeeze(tevetwi12)
        tmorntwi12 = np.squeeze(tmorntwi12)
        tmoonrise = np.squeeze(tmoonrise)
        tmoonset = np.squeeze(tmoonset)

    return midnight, tsunset, tsunrise, tevetwi12, tmorntwi12, tmoonrise, tmoonset