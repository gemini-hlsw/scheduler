#! /usr/bin/env python
"""A numpy vectorized version of thorskyutil.py -- utility and miscellaneous time and the sky
   routines built mostly on astropy.
"""

# Copyright John Thorstensen, 2018; offered under the GNU Public License 3.0
# Vectorized by Bryan Miller, Gemini Observatory

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, PrecessedGeocentric, Angle, Longitude, Distance
from astropy.coordinates import GeocentricTrueEcliptic, FK5
from astropy.time import Time, TimeDelta
from datetime import datetime, timedelta
import pkgutil


class thorconsts:
    """Constants that are used here and there.  Some are Quantities,
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


def time_rounded_to_minute(tm, sep=' ', incl_date=False, incl_day=False,
                           named_month=False, incl_year=True):
    """Takes a datetime and returns a string giving the date correctly rounded to
    nearest minute.  Calls strftime.

    Parameters
    ----------
    tm : datetime
    sep : str, default = ' '
         separator between output fields
    incl_date : boolean, default = False
         include calendar date in output
    incl_day : boolean, default = False
         include day of week in output string
    named_month : boolean, default = False
         output name of month instead of number
    incl_year : boolean, default = False
         include year in output string
    """

    # To get nearest minute, simply add half a minute to the time
    # and then format -- this works perfectly in edge cases (e.g.,
    # 2018-12-31  23:59:31  rounded to nearest minute gives 2019-01-01 00:00)

    tm += timedelta(seconds=30)
    if incl_date and incl_day:
        if incl_year:
            return tm.strftime("%a %Y-%m-%d %H:%M")
        else:
            return tm.strftime("%a %m-%d %H:%M")
    elif incl_date:
        if incl_year:
            return tm.strftime("%Y-%m-%d %H:%M")
        else:
            return tm.strftime("%m-%d %H:%M")
    elif incl_day:
        return tm.strftime("%a %H:%M")
    else:
        return tm.strftime("%H:%M")


def currentgeocentframe(time):
    """Returns a PrecessedGeocentric frame for the equinox
    spcified by the time.

    Parameters
    ----------

    time : astropy Time, if an array then the first entry is used

    Returns

    an astropy PrecessedGeocentric time.
    """
    # Generate a PrecessedGeocentric frame for the current equinox.
    time_ep = 2000. + (np.asarray(time.jd) - thorconsts.J2000) / 365.25
    if time_ep.ndim == 0:
        time_ep = time_ep[None]  # Makes 1D
    eq = Time("J{:7.2f}".format(time_ep[0]))
    # print(eq)
    return PrecessedGeocentric(equinox=eq)


def currentFK5frame(time):
    """Returns an FK5 frame for the equinox of the time.

    Parameters
    ----------

    time : astropy Time, if an array then the first entry is used

    Returns: an astropy coordinate frame.
    """
    time_ep = 2000. + (np.asarray(time.jd) - thorconsts.J2000) / 365.25
    if time_ep.ndim == 0:
        time_ep = time_ep[None]  # Makes 1D
    eq = Time("J{:7.2f}".format(time_ep[0]))

    #   print eq
    return FK5(equinox=eq)


def min_max_alt(lat, dec):
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
        # return (Angle(np.squeeze(minalt), unit = u.rad), Angle(np.squeeze(maxalt), unit = u.rad))
        minalt = np.squeeze(minalt)
        maxalt = np.squeeze(maxalt)
    return Angle(minalt, unit=u.rad), Angle(maxalt, unit=u.rad)


def ha_alt(dec, lat, alt):
    """Return an Angle giving the hour angle (from spherical astronomy, east
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


# A direct translation of the original, but the subsequent function is simpler
# def hrs_up(timeup, timedown, evening, morning) :
#     """returns a TimeDelta giving how long an object is up during the
#     night, basically the intersection between the interval is's 'up' above
#     a given altitude, and the interval when it's nighttime.
#
#     Checks are implemented for circumpolar objects that can set and come back up.
#
#     Parameters
#     ----------
#     timeup : astropy Time, can be array
#         Time at which the object rises above an altitude
#     timedown : astropy Time, can be array of the same size as timeup
#         Time at which an object sets below a given altitude
#     evening :
#         Time of the beginning of the night
#     morning :
#         Time of the ending of the night
#     """
#
#    # all are Times.
#    #  timeup - when the object rises past a given altitle
#    #  timedown - when the object sets past a given altitle
#    #  evening - time of evening twiligt (however defined)
#    #  morning - time of morning twiligt
#    # return value will be a TimeDelta
#
#     timeup = Time(np.asarray(timeup.jd), format='jd')
#     scalar_input = False
#     if timeup.ndim == 0:
#         timeup = timeup[None]
#         scalar_input = True
#     timedown = Time(np.asarray(timedown.jd), format='jd')
#     if timedown.ndim == 0:
#         timedown = timedown[None]
#
#     if len(timeup) != len(timedown):
#         print('Error: timeup and timedown must have the same dimensions')
#         return None
#
#     hrsup = TimeDelta(0.0*np.zeros(len(timeup)), format='jd')
#
#     ii = np.where(timeup < evening)[0][:] # rises before evening
#     if len(ii) != 0:
#         jj = ii[np.where(timedown[ii] > morning)[0][:]]
#         if len(jj) != 0:
#              hrsup[jj] = morning - evening
#
#         kk = ii[np.where(timedown[ii] > evening)[0][:]]
#         if len(kk) != 0:
#             # careful - circumpolar objects can come back up
#             # a second time before morning.
#             timeup2 = timeup[kk] + thorconsts.SIDDAY
#             ll = kk[np.where(timeup2 > morning)[0][:]]
#             if len(ll) != 0: # usual case - doesn't rise again.
#                 hrsup[ll] = timedown[ll] - evening
#             ll = kk[np.where(timeup2 <= morning)[0][:]]
#             if len(ll) != 0:
#                 hrsup[ll] = (timedown[ll] - evening) + (morning[ll] - timeup2)
#
#         kk = ii[np.where(timedown[ii] <= evening)[0][:]]
#         if len(kk) != 0:
#             hrsup[kk] = thorconsts.ZERO_TIMEDELTA
#
#     ii = np.where(timedown > morning)[0][:] # still up at morning twi
#     if len(ii) != 0:
#         jj = ii[np.where(timeup[ii] > morning)[0][:]]
#         if len(jj) != 0:
#             hrsup[jj] = thorconsts.ZERO_TIMEDELTA
#
#         kk = ii[np.where(timeup[ii] <= morning)[0][:]]
#         if len(kk) != 0:
#             # again an object can be up more than once per night
#             timedown0 = timedown[kk] - thorconsts.SIDDAY
#             ll = kk[np.where(timedown0 < evening)[0][:]]
#             if len(ll) != 0:
#                 hrsup[ll] = morning - timeup[ll]
#             ll = kk[np.where(timedown0 >= evening)[0][:]]
#             if len(ll) != 0:
#                 hrsup[ll] = (timedown0 - evening) + (morning - timeup[ll])
#
#     ii = np.where(np.logical_and(timeup >= evening, timedown <= morning))[0][:]  # up and down the same night.
#     if len(ii) != 0:
#         hrsup[ii] = timedown[ii] - timeup[ii]
#
#     if scalar_input:
#         hrsup = np.squeeze(hrsup)
#     return hrsup


def hrs_up(timeup, timedown, evening, morning):
    """returns a TimeDelta giving how long an object is up during the
    night, basically the intersection between the interval is's 'up' above
    a given altitude, and the interval when it's nighttime.

    Checks are implemented for circumpolar objects that can set and come back up.

    Parameters
    ----------
    timeup : astropy Time, can be array
        Time at which the object rises above an altitude
    timedown : astropy Time, can be array of the same size as timeup
        Time at which an object sets below a given altitude
    evening : astropy Time, can be array of the same size as timeup
        Time of the beginning of the night
    morning : astropy Time, can be array of the same size as timeup
        Time of the ending of the night
    """

    # all are Times.
    #  timeup - when the object rises past a given altitle
    #  timedown - when the object sets past a given altitle
    #  evening - time of evening twiligth (however defined)
    #  morning - time of morning twilight
    # return value will be a TimeDelta

    timeup = Time(np.asarray(timeup.jd), format='jd')
    scalar_input = False
    if timeup.ndim == 0:
        timeup = timeup[None]
        scalar_input = True
    timedown = Time(np.asarray(timedown.jd), format='jd')
    if timedown.ndim == 0:
        timedown = timedown[None]
    evening = Time(np.asarray(evening.jd), format='jd')
    if evening.ndim == 0:
        evening = evening[None]
    morning = Time(np.asarray(morning.jd), format='jd')
    if morning.ndim == 0:
        morning = morning[None]

    if len(timeup) != len(timedown):
        print('Error: timeup and timedown must have the same dimensions')
        return None

    hrsup = TimeDelta(0.0 * np.zeros(len(timeup)), format='jd')

    # up all night
    ii = np.where(np.logical_and(timeup < evening, timedown > morning))[0][:]
    if len(ii) != 0:
        hrsup[ii] = morning[ii] - evening[ii]

    # up and down the same night
    ii = np.where(np.logical_and(timeup >= evening, timedown <= morning))[0][:]
    if len(ii) != 0:
        # rise then set
        jj = ii[np.where(timedown[ii] >= timeup[ii])[0][:]]
        if len(jj) != 0:
            hrsup[jj] = timedown[jj] - timeup[jj]
        # set then rise the same night
        kk = ii[np.where(timeup[ii] > timedown[ii])[0][:]]
        if len(kk) != 0:
            hrsup[kk] = (timedown[kk] - evening[kk]) + (morning[kk] - timeup[kk])

    # Rise before evening
    ii = np.where(np.logical_and(timeup < evening, timedown <= morning))[0][:]
    if len(ii) != 0:
        hrsup[ii] = timedown[ii] - evening[ii]

    # Set after morning
    ii = np.where(np.logical_and(timeup >= evening, timedown > morning))[0][:]
    if len(ii) != 0:
        hrsup[ii] = morning[ii] - timeup[ii]

    if scalar_input:
        hrsup = np.squeeze(hrsup)
    return hrsup


def lpsidereal(time, location):
    """moderate-precision (1 sec) local sidereal time

    Adapted with minimal changes from skycalc routine. Native
    astropy routines are unnecessarily precise for our purposes and
    rather slow.

    Parameters :
       time : Time
       location : EarthLocation

    Returns:
       Angle   (sidereal time is the hour angle of the equinox.)

    """

    jdint = np.asarray(time.jd, dtype=int)
    scalar_input = False
    if jdint.ndim == 0:
        jdint = jdint[None]
        scalar_input = True

    jdfrac = time.jd - jdint

    jdmid = jdint + 0.5
    ut = jdfrac - 0.5
    ii = np.where(jdfrac < 0.5)[0][:]
    if len(ii) != 0:
        jdmid[ii] = jdint[ii] - 0.5
        ut[ii] = jdfrac[ii] + 0.5  # as fraction of a day.

    t = (jdmid - thorconsts.J2000) / 36525.

    sid = (24110.54841 + 8640184.812866 * t + 0.093104 * t ** 2 - 6.2e-6 * t ** 3) / 86400.
    # at Greenwich
    sid_int = sid.astype(np.int)
    sid = sid - sid_int
    # longitude is measured east so add.
    sid = sid + 1.0027379093 * ut + location.lon.hour / 24.
    sid_int = sid.astype(np.int)
    sid = (sid - sid_int) * 24.
    # if(sid < 0.) : sid = sid + 24.
    sidout = Angle(sid, unit=u.hour)
    sidout.wrap_at(24. * u.hour, inplace=True)

    if scalar_input:
        return np.squeeze(sidout)
    return sidout


# In order to avoid having the altaz transformation look up
# the USNO database, we need our own altaz transformation.

def altazparang(dec, ha, lat):
    """Compute altitude above horizon, azimuth, and parallactic angle.

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


def angle_between(a, b):
    """Find the angle between two solar system body positions with
    the cartesian representation. """
    # first draft used np.linalg but these cartesian representations are
    # not numpy arrays.
    nra = a / np.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)
    nrb = b / np.sqrt(b.x ** 2 + b.y ** 2 + b.z ** 2)
    # the clip function avoids roundoffs outside of arccos domain.
    return np.arccos(np.clip(nra.x * nrb.x + nra.y * nrb.y + nra.z * nrb.z, -1., 1.))


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


def geocent(geolong, geolat, height):
    """geocentric XYZ coordinates for a location at a longitude,
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

    denom = (1. - thorconsts.FLATTEN) * np.sin(geolat)
    denom = np.cos(geolat) * np.cos(geolat) + denom * denom
    C_geo = 1. / np.sqrt(denom)
    S_geo = (1. - thorconsts.FLATTEN) * (1. - thorconsts.FLATTEN) * C_geo
    C_geo = C_geo + height / thorconsts.EQUAT_RAD
    #  deviation from almanac notation -- include height here.
    S_geo = S_geo + height / thorconsts.EQUAT_RAD
    # distancemultiplier = Distance(thorconsts.EQUAT_RAD, unit = u.m)
    x_geo = thorconsts.EQUAT_RAD * C_geo * np.cos(geolat) * np.cos(geolong)
    y_geo = thorconsts.EQUAT_RAD * C_geo * np.cos(geolat) * np.sin(geolong)
    z_geo = thorconsts.EQUAT_RAD * S_geo * np.sin(geolat)

    return x_geo, y_geo, z_geo


def precessmatrix(t1, t2):
    """Create a matrix to transform celestial Cartesian coordinates from equinox t1 to equinox t2.

    The matrix is used for speed in precessing a large number of bright-star
    coordinates in the GUI program.

    Parameters
    ----------

    t1 : Time
       time to transform from
    t2 : Time
       time to transform to

    Returns: numpy 3x3 array

    """
    # Generates a matrix that transforms coordinates from equinox t1 to equinox
    # t2.  For fast precession of the background stars in display; the astropy
    # routines should be fast for everything else.

    # print "t1.jd, thorconsts.J2000",t1.jd,thorconsts.J2000

    ti = (t1.jd - thorconsts.J2000) / 36525.  # * u.d)
    #    print "ti",ti
    tf = (t2.jd - thorconsts.J2000) / 36525. - ti  # * u.d)
    #    print "tf",tf

    zeta = (2306.2181 + 1.39656 * ti + 0.000139 * ti ** 2) * tf + \
           (0.30188 - 0.000344 * ti) * tf ** 2 + 0.017998 * tf ** 3
    z = zeta + (0.79280 + 0.000410 * ti) * tf ** 2 + 0.000205 * tf * 3
    theta = (2004.3109 - 0.8533 * ti - 0.000217 * ti ** 2) * tf \
            - (0.42665 + 0.000217 * ti) * tf ** 2 - 0.041833 * tf ** 3

    #    print "zeta, z, theta",zeta,z,theta

    zeta = zeta / thorconsts.ARCSEC_IN_RADIAN
    z = z / thorconsts.ARCSEC_IN_RADIAN
    theta = theta / thorconsts.ARCSEC_IN_RADIAN

    cosz = np.cos(z)
    coszeta = np.cos(zeta)
    costheta = np.cos(theta)
    sinz = np.sin(z)
    sinzeta = np.sin(zeta)
    sintheta = np.sin(theta)

    #    print "cosz coszeta costheta",cosz, coszeta,costheta
    #    print "sinz sinzeta sintheta",sinz, sinzeta,sintheta

    # compute the elements of the precession matrix -- set up
    #      here as *from* standard epoch *to* input jd.
    # build a list and then reshape into an array.

    prow1 = [coszeta * cosz * costheta - sinzeta * sinz]
    prow1.append(-1. * sinzeta * cosz * costheta - coszeta * sinz)
    prow1.append(-1. * cosz * sintheta)

    prow2 = [coszeta * sinz * costheta + sinzeta * cosz]
    prow2.append(-1. * sinzeta * sinz * costheta + coszeta * cosz)
    prow2.append(-1. * sinz * sintheta)

    prow3 = [coszeta * sintheta]
    prow3.append(-1. * sinzeta * sintheta)
    prow3.append(costheta)

    precessmat = np.array([prow1, prow2, prow3])
    #    print precessmat

    return precessmat


def cel2localmatrix(lst, lat):
    """compute matrix to take celestial cartesian coordinates to topocentric
    (local observer centered) coordinates.

    Celestial coordinates are :assumed to already be in
    equinox of date and have the standard definitions


    * X toward ra=0, dec=0
    * Y toward ra = 6 hr or 90 degrees, dec = 0
    * Z toward the north celestial pole;

    Topocentric coordinates are a little unusual:

    * X toward due west
    * Y toward due south
    * Z toward the zenith.

    Parameters
    ----------

    lst : Angle
         Local sidereal time
    lat : Angle
         Geographic latitude

    Returns: numpy 3x3 array.
    """
    # Generates the matrix to take celestial cartesian coordinates --
    # assumed to be a vector with
    # x toward (ra=0,dec=0),
    # y toward (ra = 6h, dec = 0),
    # z toward north celestial pole
    # and rotate them to a frame with
    # x = due W
    # y = due S
    # z = straight up.
    # lst and lat are Angles - local sidereal time and latitude.

    coslst = np.cos(lst)
    sinlst = np.sin(lst)

    coscolat = np.sin(lat)
    sincolat = np.cos(lat)

    # rotates around polar axis so x axis points to merdian
    mat1 = np.array([[coslst, sinlst, 0.], [-1. * sinlst, coslst, 0.], [0., 0., 1.]])

    # rotates around east-west axis to rotate z axis to zenith
    mat2 = np.array([[coscolat, 0., -1. * sincolat], [0., 1., 0.], [sincolat, 0., coscolat]])

    return mat2.dot(mat1)


def cel2xyz(coord):
    """Return normalized cartesian coordinates (direction cosines).

    Parameters
    ----------
    coord : SkyCoord

    Returns:
    numpy array of length three.
    """
    # Returns direction cosines of a SkyCoord.  Weirdly, this does not seem
    # to be included in the SkyCoord class.
    # As is standard, X is toward ra,dec = 0,0; Y toward ra = 6h, dec=0;
    # and Z toward the north celestial pole.

    cosalpha = np.cos(coord.ra)
    cosdec = np.cos(coord.dec)
    sinalpha = np.sin(coord.ra)
    sindec = np.sin(coord.dec)

    return np.array([[cosalpha * cosdec], [sinalpha * cosdec], [sindec]])


def skyproject(rotmat, cartesian):
    """Rotate an array of cartesian coordinates and do a stereographic projection
    around the resulting Z-axis.

    Used to figure out where to plot the bright stars on a map of the sky
    above you at a given moment.  The rotation matrix is meant to transform
    from the input ICRS coordinates of the stars directly to the topocentric
    coordinates; the stereographic projection can then be done using array
    arithmetic without trig functions.  Use of numpy array operations makes this
    ridiculously fast.

    Parameters
    ----------

    rotmat : numpy 3x3 array
       the rotation matrix, usually from ICRS directly to topocentric XYZ
    cartesian : numpy array 3 x N (or is it N x 3?)
       array of cartesian (usually ICRS) coordinates to be transformed and projected.

    Returns

    tuple of numpy array of x coordinates, and numpy array of y coordinates
       projected onto a plane.  I believe x = 0, y = 1 is the north horizon point,
       x = 1 and y = 0 is the east horizon point.

    """

    # rotmat is a rotation matrix -- possibly including precession --
    # to transform cartesian unit vector(s) to topocentric frame.
    # cartesian is one or more column vectors in an np array.
    # Does the rotation, and then the stereographic projection for
    # plotting.

    topocentric = rotmat.dot(cartesian)  # rotate into place

    projectedx = -1. * topocentric[1] / (1 + topocentric[2])  # stereographic projection
    projectedy = -1. * topocentric[0] / (1 + topocentric[2])

    return (projectedx, projectedy)


def getbrightest():
    """Internal package routine to read the provided bright-star list for use
    in displays.

    The bright star list is an ascii list with one line per star.  It includes
    the J2000 coordinates as Cartesian triplets, which is useful later, the
    V magnitude, the name, and a hexadecimal string that encodes an RGB color
    used to render the star.  This was crudely computed from the spectral type
    listed in the bright star catalog.
    """

    # read brightest star info from a fixed-name file included in the package.
    # modified from a routine that took a file path as the argument.

    # positions are stored as Cartesian unit vectors.  Build
    # arrays of each coord to make into a big numpy array.

    X = []
    Y = []
    Z = []

    mags = []  # V magnitudes.
    colors = []  # RGB strings for rending, precomputed.
    names = []

    bytestr = pkgutil.get_data(__name__, 'cartesian_bright.dat')
    inf = bytestr.decode('ASCII').splitlines()  # this reads similarly to a file.

    # inf = open(path,"r")
    for l in inf:
        x = l.split('\t')
        X.append(float(x[0]))
        Y.append(float(x[1]))
        Z.append(float(x[2]))
        mags.append(float(x[3]))
        colors.append(x[4])
        names.append(x[5])

    positions = np.array([X, Y, Z])
    # print positions.shape
    # print positions

    return positions, mags, colors, names


def flmoon(n, nph):
    """Compute the julian day at which moon phase nph occurs in lunation n

    Good to about 2 minutes, from Jean Meeus' "Astronomical Formulae for
    Calculators", 2nd edition, Willman-Bell.

    Parameters :
    n : int
       lunation number
    nph : int
       phase to compute; 0 = new, 1 = 1st quarter, 2 = full, 3 = last quarter

    Returns:  Julian date as a float (not a Time)
    """

    # Gives jd (+- 2 min) of phase nph on lunation n.
    # Implements formulae found in Jean Meeus' *Astronomical Formulae
    # for Calculators*, 2nd edition, Willman-Bell.

    #  n, nph lunation and phase; nph = 0 new, 1 1st, 2 full, 3 last
    #  returns jd or requested phase.

    lun = float(n) + float(nph) / 4.
    T = lun / 1236.85
    jd = 2415020.75933 + 29.53058868 * lun \
         + 0.0001178 * T * T \
         - 0.000000155 * T * T * T \
         + 0.00033 * np.sin(np.deg2rad(166.56 + 132.87 * T - 0.009173 * T * T))
    M = 359.2242 + 29.10535608 * lun - 0.0000333 * T * T - 0.00000347 * T * T * T
    M = np.deg2rad(M)
    Mpr = 306.0253 + 385.81691806 * lun + 0.0107306 * T * T + 0.00001236 * T * T * T
    Mpr = np.deg2rad(Mpr)
    F = 21.2964 + 390.67050646 * lun - 0.0016528 * T * T - 0.00000239 * T * T * T
    F = np.deg2rad(F)
    if (nph == 0) or (nph == 2):  # /* new or full */
        cor = (0.1734 - 0.000393 * T) * np.sin(M) \
              + 0.0021 * np.sin(2 * M) \
              - 0.4068 * np.sin(Mpr) \
              + 0.0161 * np.sin(2 * Mpr) \
              - 0.0004 * np.sin(3 * Mpr) \
              + 0.0104 * np.sin(2 * F) \
              - 0.0051 * np.sin(M + Mpr) \
              - 0.0074 * np.sin(M - Mpr) \
              + 0.0004 * np.sin(2 * F + M) \
              - 0.0004 * np.sin(2 * F - M) \
              - 0.0006 * np.sin(2 * F + Mpr) \
              + 0.0010 * np.sin(2 * F - Mpr) \
              + 0.0005 * np.sin(M + 2 * Mpr)
        jd = jd + cor

    else:
        cor = (0.1721 - 0.0004 * T) * np.sin(M) \
              + 0.0021 * np.sin(2 * M) \
              - 0.6280 * np.sin(Mpr) \
              + 0.0089 * np.sin(2 * Mpr) \
              - 0.0004 * np.sin(3 * Mpr) \
              + 0.0079 * np.sin(2 * F) \
              - 0.0119 * np.sin(M + Mpr) \
              - 0.0047 * np.sin(M - Mpr) \
              + 0.0003 * np.sin(2 * F + M) \
              - 0.0004 * np.sin(2 * F - M) \
              - 0.0006 * np.sin(2 * F + Mpr) \
              + 0.0021 * np.sin(2 * F - Mpr) \
              + 0.0003 * np.sin(M + 2 * Mpr) \
              + 0.0004 * np.sin(M - 2 * Mpr) \
              - 0.0003 * np.sin(2 * M + Mpr)
        if nph == 1:
            cor = cor + 0.0028 - \
                  0.0004 * np.cos(M) + 0.0003 * np.cos(Mpr)
        if nph == 3:
            cor = cor - 0.0028 + \
                  0.0004 * np.cos(M) - 0.0003 * np.cos(Mpr)
        jd = jd + cor

    return jd


def phase_descr(jd):
    """a human-readable string describing the phase of the moon.

    Given an input julian date, this determines interval to nearest
    moon phase and writes a description.

    Parameters
    ----------

    jd : float  (not a Time)
         Julian date.

    Returns a string, e.g. "2.8 days before last quarter"

    """

    nlast = int((jd - 2415020.5) / 29.5307 - 1.)  # /* find current lunation */

    #    print "nlast = ",nlast

    lastnewjd = flmoon(nlast, 0)
    newjd = flmoon(nlast + 1, 0)
    kount = 0
    # count up to find appropriate lunation
    while newjd < jd and kount < 40:
        lastnewjd = newjd
        nlast = nlast + 1
        newjd = flmoon(nlast, 0)
        kount = kount + 1
    if kount > 35:  # /* oops ... didn't find it ... */
        return "Didn't find phase in print_phase!"
    else:
        x = jd - lastnewjd
        lunage = x
        nlast = nlast - 1
        lunation = nlast
        noctiles = int(x / 3.69134)  # 3.69134 = 1/8 month; truncate.
        if noctiles == 0:
            return "{:3.1f} days since new moon".format(x), lunage, lunation
        elif noctiles <= 2:  # /* nearest first quarter */
            fqjd = flmoon(nlast, 1)
            x = jd - fqjd
            if (x < 0.):
                return "{:3.1f} days before first quarter".format(-1. * x), lunage, lunation
            else:
                return "{:3.1f} days since first quarter".format(x), lunage, lunation
        elif noctiles <= 4:  # /* nearest full */
            fljd = flmoon(nlast, 2)
            x = jd - fljd
            if (x < 0.):
                return "{:3.1f} days until full moon".format(-1. * x), lunage, lunation
            else:
                return "{:3.1f} days after full moon".format(x), lunage, lunation

        elif (noctiles <= 6):
            lqjd = flmoon(nlast, 3)
            x = jd - lqjd
            if (x < 0.):
                return "{:3.1f} days before last quarter".format(-1. * x), lunage, lunation
            else:
                return "{:3.1f} days after last quarter".format(x), lunage, lunation

        else:
            return "{:3.1f} days before new moon".format(newjd - jd), lunage, lunation


# def lpmoon(time,lat,sid) :
def lpmoon(time, obs):
    """low-precision moon position.

    Good to about 0.1 deg, from the 1992 Astronomical Almanac, p. D46.
    Note that input time is a float.

    Parameters
    ----------
    time : Time
        Time expressed as a Julian Date
    obs :  EarthLocation
        location of observatory.

    Returns

    SkyCoord with a distance set, topocentric.

    """

    # jd is a number; lat and sid are Angles.
    # hands back topocentric RA, dec (radians) and distance
    # (earth radii).

    #  Implements "low precision" moon sky_brightness from
    #  Astronomical Almanac (p. D46 in 1992 version).  Does

    jd = np.asarray(time.jd)
    scalar_input = False
    if jd.ndim == 0:
        jd = jd[None]
        scalar_input = True

    T = (jd - thorconsts.J2000) / 36525.  # jul cent. since J2000.0

    sid = lpsidereal(time, obs)
    lat = obs.lat

    lambd = 218.32 + 481267.883 * T \
            + 6.29 * np.sin(np.deg2rad(134.9 + 477198.85 * T)) \
            - 1.27 * np.sin(np.deg2rad(259.2 - 413335.38 * T)) \
            + 0.66 * np.sin(np.deg2rad(235.7 + 890534.23 * T)) \
            + 0.21 * np.sin(np.deg2rad(269.9 + 954397.70 * T)) \
            - 0.19 * np.sin(np.deg2rad(357.5 + 35999.05 * T)) \
            - 0.11 * np.sin(np.deg2rad(186.6 + 966404.05 * T))
    lambd = np.deg2rad(lambd)
    beta = 5.13 * np.sin(np.deg2rad(93.3 + 483202.03 * T)) \
           + 0.28 * np.sin(np.deg2rad(228.2 + 960400.87 * T)) \
           - 0.28 * np.sin(np.deg2rad(318.3 + 6003.18 * T)) \
           - 0.17 * np.sin(np.deg2rad(217.6 - 407332.20 * T))
    beta = np.deg2rad(beta)
    pie = 0.9508 + 0.0518 * np.cos(np.deg2rad(134.9 + 477198.85 * T)) \
          + 0.0095 * np.cos(np.deg2rad(259.2 - 413335.38 * T)) \
          + 0.0078 * np.cos(np.deg2rad(235.7 + 890534.23 * T)) \
          + 0.0028 * np.cos(np.deg2rad(269.9 + 954397.70 * T))
    pie = np.deg2rad(pie)
    distance = 1. / np.sin(pie)

    l = np.cos(beta) * np.cos(lambd)
    m = 0.9175 * np.cos(beta) * np.sin(lambd) - 0.3978 * np.sin(beta)
    n = 0.3978 * np.cos(beta) * np.sin(lambd) + 0.9175 * np.sin(beta)

    x = l * distance
    y = m * distance
    z = n * distance  # /* for topocentric correction */
    # these are Angles so no need.
    # rad_lat = lat / DEG_IN_RADIAN
    # rad_lst = sid / HRS_IN_RADIAN
    x = x - np.cos(lat) * np.cos(sid)
    y = y - np.cos(lat) * np.sin(sid)
    z = z - np.sin(lat)

    topo_dist = np.sqrt(x * x + y * y + z * z)

    l = x / topo_dist
    m = y / topo_dist
    n = z / topo_dist

    alpha = np.arctan2(m, l)
    delta = np.arcsin(n)
    distancemultiplier = Distance(thorconsts.EQUAT_RAD, unit=u.m)

    fr = currentgeocentframe(time)

    if scalar_input:
        alpha = np.squeeze(alpha)
        delta = np.squeeze(delta)
        topo_dist = np.squeeze(topo_dist)
    return SkyCoord(alpha, delta, topo_dist * distancemultiplier, frame=fr)


#
def accumoon(time, obs):  # ,geolat,lst) :
    """compute topocentric location and distance of moon to better accuracy.

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

    time_ttjd = np.asarray(time.tt.jd)  # important to use correct time argument for this!!
    scalar_input = False
    if time_ttjd.ndim == 0:
        time_ttjd = time_ttjd[None]  # Makes 1D
        scalar_input = True

    T = (time_ttjd - 2415020.) / 36525.  # this based around 1900 ... */
    Tsq = T * T
    Tcb = Tsq * T

    Lpr = 270.434164 + 481267.8831 * T - 0.001133 * Tsq + 0.0000019 * Tcb
    M = 358.475833 + 35999.0498 * T - 0.000150 * Tsq - 0.0000033 * Tcb
    Mpr = 296.104608 + 477198.8491 * T + 0.009192 * Tsq + 0.0000144 * Tcb
    D = 350.737486 + 445267.1142 * T - 0.001436 * Tsq + 0.0000019 * Tcb
    F = 11.250889 + 483202.0251 * T - 0.003211 * Tsq - 0.0000003 * Tcb
    Om = 259.183275 - 1934.1420 * T + 0.002078 * Tsq + 0.0000022 * Tcb

    Lpr = Lpr % 360.
    Mpr = Mpr % 360.
    M = M % 360.
    D = D % 360.
    F = F % 360.
    Om = Om % 360.

    sinx = np.sin(np.deg2rad(51.2 + 20.2 * T))
    Lpr = Lpr + 0.000233 * sinx
    M = M - 0.001778 * sinx
    Mpr = Mpr + 0.000817 * sinx
    D = D + 0.002011 * sinx

    sinx = 0.003964 * np.sin(np.deg2rad(346.560 + 132.870 * T - 0.0091731 * Tsq))

    Lpr = Lpr + sinx
    Mpr = Mpr + sinx
    D = D + sinx
    F = F + sinx

    sinx = np.sin(np.deg2rad(Om))
    Lpr = Lpr + 0.001964 * sinx
    Mpr = Mpr + 0.002541 * sinx
    D = D + 0.001964 * sinx
    F = F - 0.024691 * sinx
    F = F - 0.004328 * np.sin(np.deg2rad(Om + 275.05 - 2.30 * T))

    e = 1 - 0.002495 * T - 0.00000752 * Tsq

    M = np.deg2rad(M)  # these will all be arguments ... */
    Mpr = np.deg2rad(Mpr)
    D = np.deg2rad(D)
    F = np.deg2rad(F)

    lambd = Lpr + 6.288750 * np.sin(Mpr) \
            + 1.274018 * np.sin(2 * D - Mpr) \
            + 0.658309 * np.sin(2 * D) \
            + 0.213616 * np.sin(2 * Mpr) \
            - e * 0.185596 * np.sin(M) \
            - 0.114336 * np.sin(2 * F) \
            + 0.058793 * np.sin(2 * D - 2 * Mpr) \
            + e * 0.057212 * np.sin(2 * D - M - Mpr) \
            + 0.053320 * np.sin(2 * D + Mpr) \
            + e * 0.045874 * np.sin(2 * D - M) \
            + e * 0.041024 * np.sin(Mpr - M) \
            - 0.034718 * np.sin(D) \
            - e * 0.030465 * np.sin(M + Mpr) \
            + 0.015326 * np.sin(2 * D - 2 * F) \
            - 0.012528 * np.sin(2 * F + Mpr) \
            - 0.010980 * np.sin(2 * F - Mpr) \
            + 0.010674 * np.sin(4 * D - Mpr) \
            + 0.010034 * np.sin(3 * Mpr) \
            + 0.008548 * np.sin(4 * D - 2 * Mpr) \
            - e * 0.007910 * np.sin(M - Mpr + 2 * D) \
            - e * 0.006783 * np.sin(2 * D + M) \
            + 0.005162 * np.sin(Mpr - D)

    #       /* And furthermore.....*/

    lambd = lambd + e * 0.005000 * np.sin(M + D) \
            + e * 0.004049 * np.sin(Mpr - M + 2 * D) \
            + 0.003996 * np.sin(2 * Mpr + 2 * D) \
            + 0.003862 * np.sin(4 * D) \
            + 0.003665 * np.sin(2 * D - 3 * Mpr) \
            + e * 0.002695 * np.sin(2 * Mpr - M) \
            + 0.002602 * np.sin(Mpr - 2 * F - 2 * D) \
            + e * 0.002396 * np.sin(2 * D - M - 2 * Mpr) \
            - 0.002349 * np.sin(Mpr + D) \
            + e * e * 0.002249 * np.sin(2 * D - 2 * M) \
            - e * 0.002125 * np.sin(2 * Mpr + M) \
            - e * e * 0.002079 * np.sin(2 * M) \
            + e * e * 0.002059 * np.sin(2 * D - Mpr - 2 * M) \
            - 0.001773 * np.sin(Mpr + 2 * D - 2 * F) \
            - 0.001595 * np.sin(2 * F + 2 * D) \
            + e * 0.001220 * np.sin(4 * D - M - Mpr) \
            - 0.001110 * np.sin(2 * Mpr + 2 * F) \
            + 0.000892 * np.sin(Mpr - 3 * D) \
            - e * 0.000811 * np.sin(M + Mpr + 2 * D) \
            + e * 0.000761 * np.sin(4 * D - M - 2 * Mpr) \
            + e * e * 0.000717 * np.sin(Mpr - 2 * M) \
            + e * e * 0.000704 * np.sin(Mpr - 2 * M - 2 * D) \
            + e * 0.000693 * np.sin(M - 2 * Mpr + 2 * D) \
            + e * 0.000598 * np.sin(2 * D - M - 2 * F) \
            + 0.000550 * np.sin(Mpr + 4 * D) \
            + 0.000538 * np.sin(4 * Mpr) \
            + e * 0.000521 * np.sin(4 * D - M) \
            + 0.000486 * np.sin(2 * Mpr - D)

    B = 5.128189 * np.sin(F) \
        + 0.280606 * np.sin(Mpr + F) \
        + 0.277693 * np.sin(Mpr - F) \
        + 0.173238 * np.sin(2 * D - F) \
        + 0.055413 * np.sin(2 * D + F - Mpr) \
        + 0.046272 * np.sin(2 * D - F - Mpr) \
        + 0.032573 * np.sin(2 * D + F) \
        + 0.017198 * np.sin(2 * Mpr + F) \
        + 0.009267 * np.sin(2 * D + Mpr - F) \
        + 0.008823 * np.sin(2 * Mpr - F) \
        + e * 0.008247 * np.sin(2 * D - M - F) \
        + 0.004323 * np.sin(2 * D - F - 2 * Mpr) \
        + 0.004200 * np.sin(2 * D + F + Mpr) \
        + e * 0.003372 * np.sin(F - M - 2 * D) \
        + 0.002472 * np.sin(2 * D + F - M - Mpr) \
        + e * 0.002222 * np.sin(2 * D + F - M) \
        + e * 0.002072 * np.sin(2 * D - F - M - Mpr) \
        + e * 0.001877 * np.sin(F - M + Mpr) \
        + 0.001828 * np.sin(4 * D - F - Mpr) \
        - e * 0.001803 * np.sin(F + M) \
        - 0.001750 * np.sin(3 * F) \
        + e * 0.001570 * np.sin(Mpr - M - F) \
        - 0.001487 * np.sin(F + D) \
        - e * 0.001481 * np.sin(F + M + Mpr) \
        + e * 0.001417 * np.sin(F - M - Mpr) \
        + e * 0.001350 * np.sin(F - M) \
        + 0.001330 * np.sin(F - D) \
        + 0.001106 * np.sin(F + 3 * Mpr) \
        + 0.001020 * np.sin(4 * D - F) \
        + 0.000833 * np.sin(F + 4 * D - Mpr)
    #     /* not only that, but */
    B = B + 0.000781 * np.sin(Mpr - 3 * F) \
        + 0.000670 * np.sin(F + 4 * D - 2 * Mpr) \
        + 0.000606 * np.sin(2 * D - 3 * F) \
        + 0.000597 * np.sin(2 * D + 2 * Mpr - F) \
        + e * 0.000492 * np.sin(2 * D + Mpr - M - F) \
        + 0.000450 * np.sin(2 * Mpr - F - 2 * D) \
        + 0.000439 * np.sin(3 * Mpr - F) \
        + 0.000423 * np.sin(F + 2 * D + 2 * Mpr) \
        + 0.000422 * np.sin(2 * D - F - 3 * Mpr) \
        - e * 0.000367 * np.sin(M + F + 2 * D - Mpr) \
        - e * 0.000353 * np.sin(M + F + 2 * D) \
        + 0.000331 * np.sin(F + 4 * D) \
        + e * 0.000317 * np.sin(2 * D + F - M + Mpr) \
        + e * e * 0.000306 * np.sin(2 * D - 2 * M - F) \
        - 0.000283 * np.sin(Mpr + 3 * F)

    om1 = 0.0004664 * np.cos(np.deg2rad(Om))
    om2 = 0.0000754 * np.cos(np.deg2rad(Om + 275.05 - 2.30 * T))

    beta = B * (1. - om1 - om2)

    pie = 0.950724 + 0.051818 * np.cos(Mpr) \
          + 0.009531 * np.cos(2 * D - Mpr) \
          + 0.007843 * np.cos(2 * D) \
          + 0.002824 * np.cos(2 * Mpr) \
          + 0.000857 * np.cos(2 * D + Mpr) \
          + e * 0.000533 * np.cos(2 * D - M) \
          + e * 0.000401 * np.cos(2 * D - M - Mpr) \
          + e * 0.000320 * np.cos(Mpr - M) \
          - 0.000271 * np.cos(D) \
          - e * 0.000264 * np.cos(M + Mpr) \
          - 0.000198 * np.cos(2 * F - Mpr) \
          + 0.000173 * np.cos(3 * Mpr) \
          + 0.000167 * np.cos(4 * D - Mpr) \
          - e * 0.000111 * np.cos(M) \
          + 0.000103 * np.cos(4 * D - 2 * Mpr) \
          - 0.000084 * np.cos(2 * Mpr - 2 * D) \
          - e * 0.000083 * np.cos(2 * D + M) \
          + 0.000079 * np.cos(2 * D + 2 * Mpr) \
          + 0.000072 * np.cos(4 * D) \
          + e * 0.000064 * np.cos(2 * D - M + Mpr) \
          - e * 0.000063 * np.cos(2 * D + M - Mpr) \
          + e * 0.000041 * np.cos(M + D) \
          + e * 0.000035 * np.cos(2 * Mpr - M) \
          - 0.000033 * np.cos(3 * Mpr - 2 * D) \
          - 0.000030 * np.cos(Mpr + D) \
          - 0.000029 * np.cos(2 * F - 2 * D) \
          - e * 0.000029 * np.cos(2 * Mpr + M) \
          + e * e * 0.000026 * np.cos(2 * D - 2 * M) \
          - 0.000023 * np.cos(2 * F - 2 * D + Mpr) \
          + e * 0.000019 * np.cos(4 * D - M - Mpr);

    beta = Angle(np.deg2rad(beta), unit=u.rad)
    lambd = Angle(np.deg2rad(lambd), unit=u.rad)

    dist = Distance(1. / np.sin(np.deg2rad(pie)) * thorconsts.EQUAT_RAD)

    # place these in a skycoord in ecliptic coords of date.  Handle distance
    # separately since it does not transform properly for some reason.

    eq = 'J{:7.2f}'.format(2000. + (time_ttjd[0] - thorconsts.J2000) / 365.25)
    fr = GeocentricTrueEcliptic(equinox=eq)
    inecl = SkyCoord(lon=Angle(lambd, unit=u.rad), lat=Angle(beta, unit=u.rad), frame=fr)

    # Transform into geocentric equatorial.

    geocen = inecl.transform_to(currentgeocentframe(time))

    # Do the topo correction yourself. First form xyz coords in equatorial syst of date

    x = dist * np.cos(geocen.ra) * np.cos(geocen.dec)
    y = dist * np.sin(geocen.ra) * np.cos(geocen.dec)
    z = dist * np.sin(geocen.dec)

    # Now compute geocentric location of the observatory in a frame aligned with the
    # equatorial system of date, which one can do simply by replacing the west longitude
    # with the sidereal time

    # Exact match with thorskyutil/skycalc with the line below
    # xobs, yobs, zobs = geocent(lpsidereal(time, obs), obs.lat, 2000.*u.m) # don't have obs.height yet
    xobs, yobs, zobs = geocent(lpsidereal(time, obs), obs.lat, obs.height)

    # recenter moon's cartesian coordinates on position of obs

    x = x - xobs
    y = y - yobs
    z = z - zobs

    # form the toposcentric ra and dec and bundle them into a skycoord of epoch of date.

    topodist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    raout = np.arctan2(y, x)
    decout = np.arcsin(z / topodist)
    # topocen = SkyCoord(raout, decout, unit = u.rad, frame = currentgeocentframe(time))

    if scalar_input:
        raout = np.squeeze(raout)
        decout = np.squeeze(decout)
        topodist = np.squeeze(topodist)
    return SkyCoord(raout, decout, unit=u.rad, frame=currentgeocentframe(time)), topodist


def lpsun(time):
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

    n = jd - thorconsts.J2000  # referred to J2000
    L = 280.460 + 0.9856474 * n
    g = np.deg2rad(357.528 + 0.9856003 * n)
    lambd = np.deg2rad(L + 1.915 * np.sin(g) + 0.020 * np.sin(2. * g))
    epsilon = np.deg2rad(23.439 - 0.0000004 * n)

    x = np.cos(lambd)
    y = np.cos(epsilon) * np.sin(lambd)
    z = np.sin(epsilon) * np.sin(lambd)

    ra = np.arctan2(y, x)
    dec = np.arcsin(z)

    fr = currentgeocentframe(time)

    if scalar_input:
        ra = np.squeeze(ra)
        dec = np.squeeze(dec)
    return SkyCoord(ra, dec, frame=fr, unit='radian')


def jd_sun_alt(alt, tguess, location):
    """time at which the sun crosses a given elevation.

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

    sunpos = lpsun(tguess)
    # print "sunpos entering",sunpos
    # print "tguess.jd, longit:",tguess.jd, location.lon.hour
    tolerance = Angle(1.0e-4, unit=u.rad)

    delt = TimeDelta(0.002, format='jd')  # timestep
    # print "sidereal: ",lpsidereal(tguess, location)
    # print "sunpos.ra: ",sunpos.ra

    ha = lpsidereal(tguess, location) - sunpos.ra
    # print "ha entering",ha
    alt2, az, parang = altazparang(sunpos.dec, Angle(ha, unit=u.hourangle), location.lat)
    # print "alt2",alt2
    tguess = tguess + delt
    sunpos = lpsun(tguess)
    # print "sunpos with delt",sunpos
    alt3, az, parang = altazparang(sunpos.dec, lpsidereal(tguess, location) - sunpos.ra, location.lat)
    err = alt3 - alt
    # print "alt3, alt, err",alt3,alt,err
    deriv = (alt3 - alt2) / delt
    # print "deriv",deriv
    kount = np.zeros(len(tguess), dtype=int)
    kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]
    while (len(kk) != 0):
        tguess[kk] = tguess[kk] - err[kk] / deriv[kk]
        sunpos = None
        sunpos = lpsun(tguess[kk])
        alt3[kk], az[kk], parang[kk] = altazparang(sunpos.dec, lpsidereal(tguess[kk], location) - sunpos.ra,
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
    """Time at which moon passes a given altitude.

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

    moonpos, topodist = accumoon(tguess, location)
    # print "npos entering",moonpos
    # print "tguess.jd, longit:",tguess.jd, location.lon.hour
    tolerance = Angle(1.0e-4, unit=u.rad)

    delt = TimeDelta(0.002, format='jd')  # timestep
    # print "sidereal: ",lpsidereal(tguess, location)
    # print "moonpos.ra: ",moonpos.ra

    ha = lpsidereal(tguess, location) - moonpos.ra
    # print "ha entering",ha
    alt2, az, parang = altazparang(moonpos.dec, Angle(ha, unit=u.hourangle), location.lat)
    # print "alt2",alt2
    tguess = tguess + delt
    moonpos, topodist = accumoon(tguess, location)
    # print "moonpos with delt",moonpos
    alt3, az, parang = altazparang(moonpos.dec, lpsidereal(tguess, location) - moonpos.ra, location.lat)
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
        moonpos, topodist = accumoon(tguess[kk], location)
        alt3[kk], az[kk], parang[kk] = altazparang(moonpos.dec, lpsidereal(tguess[kk], location) - moonpos.ra,
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


def local_midnight_Time(aTime, localtzone):
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


def ztwilight(alt):
    """Estimate twilight contribution to zenith sky brightness, in magnitudes
    per square arcsecond.

    Evaluates a polynomial approximation to observational data (see source for
    reference) of zenith sky brightness (blue) as a function of the sun's elevation
    from -0.9 degrees to -18 degrees.  For reference, 3 mag is roughly Nautical
    twilight and looks 'pretty dark'; something like 10 mag is about the maximum
    for broadband sky flats in many cases.

    Parameters
    ----------
    alt : Angle
        Sun's elevation.  Meaningful range is -0.9 to -18 degrees.

    Returns
        float.  If the sun is up, returns 20; if the sun below -18, returns 0.

    """
    # Given an Angle alt in the range -0.9 to -18 degrees,
    # evaluates a polynomial expansion for the approximate brightening
    # in magnitudes of the zenith in twilight compared to its
    # value at full night, as function of altitude of the sun (in degrees).
    # To get this expression I looked in Meinel, A.,
    # & Meinel, M., "Sunsets, Twilight, & Evening Skies", Cambridge U.
    # Press, 1983; there's a graph on p. 38 showing the decline of
    # zenith twilight.  I read points off this graph and fit them with a
    # polynomial.
    # Comparison with Ashburn, E. V. 1952, JGR, v.57, p.85 shows that this
    # is a good fit to his B-band measurements.

    alt = Angle(np.asarray(alt.deg), unit=u.deg)
    scalar_input = False
    if alt.ndim == 0:
        alt = alt[None]  # Makes 1D
        scalar_input = True

    twisb = np.zeros(len(alt))
    # flag for sun up, also not grossly wrong.
    ii = np.where(alt.deg > -0.9)[0][:]
    if len(ii) != 0:
        twisb[ii] = 20.

    # fully dark, no contrib to zenith skyglow.
    jj = np.where(alt.deg < -18.)[0][:]
    if len(jj) != 0:
        twisb[jj] = 0.

    kk = np.where(np.logical_and(alt.deg >= -18., alt.deg <= -0.9))[0][:]
    if len(kk) != 0:
        y = (-1. * alt[kk].deg - 9.0) / 9.0  # my polynomial's argument
        twisb[kk] = ((2.0635175 * y + 1.246602) * y - 9.4084495) * y + 6.132725

    if scalar_input:
        twisb = np.squeeze(twisb)
    return twisb


def nightevents(aTime, location, localtzone, verbose=True):
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

    midnight = local_midnight_Time(aTime, localtzone)  # nearest clock-time midnight (UT)
    # print("midnight:", midnight)
    lstmid = lpsidereal(midnight, location)

    # Sun
    sunmid = lpsun(midnight)
    sunpos = lpsun(aTime)

    # allow separate rise and set altitudes for horizon effects
    horiz = (-0.883 - np.sqrt(2. * location.height / thorconsts.EQUAT_RAD) * (180. / np.pi)) * u.deg
    # print(horiz)
    setalt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    risealt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    twialt12 = Angle(-12. * np.ones(nt), unit=u.deg)  # 12 degree nautical twilight

    # print(sunmid.dec, setalt)
    sunsetha = ha_alt(sunmid.dec, location.lat, setalt)  # corresponding hr angles
    sunriseha = Angle(2. * np.pi, unit=u.rad) - ha_alt(sunmid.dec, location.lat, risealt)  # corresponding hr angles
    twilightha12 = ha_alt(sunmid.dec, location.lat, twialt12)

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
    tsunset = jd_sun_alt(setalt, tsunset, location)

    tsunrise = midnight + TDsunrise  # first approx
    tsunrise = jd_sun_alt(risealt, tsunrise, location)

    tevetwi12 = midnight - TDevetwi12
    tevetwi12 = jd_sun_alt(twialt12, tevetwi12, location)

    tmorntwi12 = midnight + TDmorntwi12
    tmorntwi12 = jd_sun_alt(twialt12, tmorntwi12, location)

    if verbose:
        print("sunset: ", tsunset)
        print("sunrise: ", tsunrise)
        print("eve twi12: ", tevetwi12)
        print("morn twi12:", tmorntwi12)
    # Moon
    moonmid = lpmoon(midnight, location)
    hamoonmid = lstmid - moonmid.ra
    hamoonmid.wrap_at(12. * u.hour, inplace=True)
    moonpos, topodist = accumoon(aTime, location)

    if verbose:
        print("moon at midnight: ", moonmid.to_string('hmsdms'))
        print("hamoonmid: ", hamoonmid.hour, 'hr')

    # roughlunarday = TimeDelta(1.0366, format='jd')

    moonsetha = ha_alt(moonmid.dec, location.lat, setalt)  # corresponding hr angles
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

    moonriseha = -1. * ha_alt(moonmid.dec, location.lat, risealt)  # signed
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


def xair(zd):
    """
    Evaluate true airmass, equation 3 from Krisciunas &  Schaefer 1991
    zd is a Quantity

    zd : '~astropy.units.Quantity'
        Float or numpy array of zenith distance angles

    Trick for handling arrays and scalars from
    https://stackoverflow.com/questions/29318459/python-function-that-handles-scalar-or-arrays
    """

    zd = np.asarray(zd.to_value(u.rad).data) * u.rad
    scalar_input = False
    if zd.ndim == 0:
        zd = zd[None]  # Makes 1D
        scalar_input = True

    x = np.zeros(len(zd))

    # below the horizon
    kk = np.where(zd > 90. * u.deg)[0][:]
    if len(kk) != 0:
        x[kk] = 10.

    # above the horizon
    ik = np.where(zd <= 90. * u.deg)[0][:]
    if len(ik) != 0:
        v = np.sqrt(1.0 - 0.96 * np.sin(zd[ik]) ** 2)
        ii = np.where(v != 0.0)[0][:]
        if len(ii) != 0:
            x[ik[ii]] = 1. / v[ii]
        jj = np.where(v == 0.0)[0][:]
        if len(jj) != 0:
            x[ik[jj]] = 10.

    if scalar_input:
        return np.squeeze(x)
    return x


def lunskybright(alpha, rho, kzen, altmoon, alt, moondist, sunalt):
    """Evaluate sky brightness due to the moon, in V mag per square
     arcsec.

     Expressions are from  K. Krisciunas and B. E. Schaeffer (1991)
     PASP 103, 1033.
     For comparison, 'dark' night sky is about 21.5 mag per square
     arcsec.  Looking well away from the moon on a clear night with
     full moon, one can expect roughly 18th mag per square arcsec.

     Parameters
     ----------
     alpha : Angle
             angle subtended by sun and moon at observer, converted
             internally to its supplement.
         rho : Angle
             angular distance of object from the moon.
         kzen : float
             zenight extinction coefficient (typically 0.15 or so for V)
         altmoon : Angle
             altitude of moon above horizon
         alt : Angle
             altitude of object above horizon
         moondist : Quantity
             distance to moon as an astropy Quantity

         Returns

         float

    """

    # Evaluates predicted LUNAR part of sky brightness, in
    #  V magnitudes per square arcsecond, following K. Krisciunas
    #  and B. E. Schaeffer (1991) PASP 103, 1033.
    #
    #  alpha = separation of sun and moon as seen from earth,
    #  converted internally to its supplement,
    #  rho = separation of moon and object,
    #  kzen = zenith extinction coefficient,
    #  altmoon = altitude of moon above horizon,
    #  alt = altitude of object above horizon
    #  moondist = distance to moon, in earth radii
    #
    #  all the angles are Angles.

    # def lunskybright(alpha, rho, kzen, altmoon, alt, moondist, sunalt):

    alpha = Angle(np.asarray(alpha.deg), unit=u.deg)
    rho = Angle(np.asarray(rho.deg), unit=u.deg)
    altmoon = Angle(np.asarray(altmoon.rad), unit=u.rad)
    alt = Angle(np.asarray(alt.rad), unit=u.rad)
    moondist = u.Quantity(np.asarray(moondist.value), u.m)
    scalar_input = False
    if alpha.ndim == 0:
        alpha = alpha[None]  # Makes 1D
        rho = rho[None]
        altmoon = altmoon[None]
        alt = alt[None]
        moondist = moondist[None]
        scalar_input = True

    # print(alpha, rho, altmoon, alt, moondist)

    skybright = np.zeros(len(alpha))

    # Will avoid this calculation if inappropriate, but for safety
    # return nonsense values if
    # if altmoon < 0. : return 99.
    # moon is down
    ii = np.where(altmoon < 0. * u.deg)[0][:]
    if len(ii) != 0:
        skybright[ii] = 99.

    # if alt < 0. : return 99.
    # object is down
    ii = np.where(alt < 0. * u.deg)[0][:]
    if len(ii) != 0:
        skybright[ii] = 99.

    # if sunalt > -10. * u.deg : return 99.
    # bright twilight or daylight.
    ii = np.where(sunalt > -10. * u.deg)[0][:]
    if len(ii) != 0:
        skybright[ii] = 99.

    im = np.where(np.logical_and(altmoon >= 0. * u.deg,
                                 np.logical_and(alt > 0. * u.deg, sunalt <= -10 * u.deg)))[0][:]
    if len(im) != 0:
        Bmoon = np.zeros(len(im))
        alpham = Angle(180. * u.deg) - alpha[im]
        Zmoon = Angle(90. * u.deg) - altmoon[im]
        Z = Angle(90. * u.deg) - alt[im]
        norm_moondist = moondist[im] / (60.27 * thorconsts.EQUAT_RAD)  # divide by mean distance

        istar = -0.4 * (3.84 + 0.026 * abs(alpham.deg) + 4.0e-9 * alpham.deg ** 4)  # eqn 20
        #     print "istar",istar, "norm_moondist",norm_moondist
        istar = (10. ** istar) / (norm_moondist ** 2.)
        #     print "istar post exp",istar,"alpha_deg",alpha.deg
        #     if abs(alpha.deg) < 7. :
        # crude accounting for opposition effect
        hh = np.where(abs(alpham) < 7. * u.deg)[0][:]
        if len(hh) != 0:
            istar[hh] = istar[hh] * (1.35 - 0.05 * abs(alpham[hh].deg))
        # istar = istar * (1.35 - 0.05 * abs(istar)) # original, bug
        # 35 per cent brighter at full, effect tapering linearly to
        #   zero at 7 degrees away from full. mentioned peripherally in
        #   Krisciunas and Scheafer, p. 1035. */

        # print "rho, cos(rho)",rho.rad,np.cos(rho)

        fofrho = 229087. * (1.06 + np.cos(rho[im]) ** 2.)
        # if abs(rho.deg) > 10.:
        ii = np.where(rho[im] > 10. * u.deg)[0][:]
        if len(ii) != 0:
            fofrho[ii] = fofrho[ii] + 10. ** (6.15 - rho[im[ii]].deg / 40.)  # /* eqn 21 */\

        jj = np.where(np.logical_and(abs(rho[im]) > 0.25 * u.deg,
                                     abs(rho[im]) <= 10. * u.deg))[0][:]
        if len(jj) != 0:
            fofrho[jj] = fofrho[jj] + 6.2e7 / (rho[im[jj]].deg ** 2)  # eq 19

        kk = np.where(abs(rho[im]) <= 0.25 * u.deg)[0][:]
        if len(kk) != 0:
            fofrho[kk] = fofrho[kk] + 9.9e8  # for 1/4 degree -- radius of moon! */

        # print "fofrho",fofrho

        Xzm = xair(Zmoon)
        # print "Xzm",Xzm

        Xo = xair(Z)
        # print "Xo",Xo

        Bmoon = fofrho * istar * 10. ** (-0.4 * kzen * Xzm) \
                * (1. - 10. ** (-0.4 * kzen * Xo))  # /* nanoLamberts */

        # print "Bmoon",Bmoon

        # if Bmoon > 0.001:
        ij = np.where(Bmoon > 0.001)[0][:]
        if len(ij) != 0:
            skybright[im[ij]] = 22.50 - 1.08574 * np.log(Bmoon[ij] / 34.08)  # /* V mag per sq arcs-eqn 1 */

        ik = np.where(Bmoon <= 0.001)[0][:]
        if len(ik) != 0:
            skybright[im[ik]] = 99.

    if scalar_input:
        return np.squeeze(skybright)
    return skybright



