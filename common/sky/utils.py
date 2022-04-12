from typing import Tuple, Union
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time, TimezoneInfo
from astropy.coordinates import PrecessedGeocentric, Angle, EarthLocation
import numpy as np
import numpy.typing as npt
from common.sky.constants import J2000, FLATTEN, EQUAT_RAD
from common.sky.altitude import AngleParam
from datetime import datetime, timedelta
from astropy.coordinates import BaseRADecFrame


def current_geocent_frame(time: Time) -> BaseRADecFrame:
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
    time_ep = 2000. + (np.asarray(time.jd) - J2000) / 365.25
    if time_ep.ndim == 0:
        time_ep = time_ep[None]  # Makes 1D
    equinox = Time(f'J{time_ep[0]:7.2f}')

    return PrecessedGeocentric(equinox=equinox)


def geocentric_coors(geolong: Angle, geolat: float, height: float) -> Tuple[float, float, float]:
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

    denom = (1. - FLATTEN) * np.sin(geolat)
    denom = np.cos(geolat) * np.cos(geolat) + denom * denom
    C_geo = 1. / np.sqrt(denom)
    S_geo = (1. - FLATTEN) * (1. - FLATTEN) * C_geo
    C_geo = C_geo + height / EQUAT_RAD
    #  deviation from almanac notation -- include height here.
    S_geo = S_geo + height / EQUAT_RAD
    # distancemultiplier = Distance(_Constants.EQUAT_RAD, unit = u.m)
    x_geo = EQUAT_RAD * C_geo * np.cos(geolat) * np.cos(geolong)
    y_geo = EQUAT_RAD * C_geo * np.cos(geolat) * np.sin(geolong)
    z_geo = EQUAT_RAD * S_geo * np.sin(geolat)

    return x_geo, y_geo, z_geo


def min_max_alt(lat: Angle, dec: AngleParam) -> Tuple[Angle, Angle]:
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


def local_midnight_time(time: Time, localtzone: TimezoneInfo) -> Time:
    """find nearest local midnight (UT).

    If it's before noon local time, returns previous midnight;
    if after noon, return next midnight.

    Parameters :

    time : astropy Time

    localtzone : timezone object.

    Returns

    Time.  This is not zone-aware, but should be correct.
    """

    # takes an astropy Time and the local time zone and
    # generates another Time which is the nearest local
    # clock-time midnight.  The returned Time is unaware
    # of the timezone but should be correct.

    time = Time(np.asarray(time.iso), format='iso')
    scalar_input = False
    if time.ndim == 0:
        time = time[None]  # Makes 1D
        scalar_input = True

    datetmid = []
    for time in time:
        datet = time.to_datetime(timezone=localtzone)

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
    mid = julian_int + 0.5
    ut = fraction - 0.5
    less_than_half = np.where(fraction < 0.5)[0][:]
    if len(less_than_half) != 0:
        mid[less_than_half] = julian_int[less_than_half] - 0.5
        ut[less_than_half] = fraction[less_than_half] + 0.5  # as fraction of a day.
    
    t = (mid - J2000) / 36525.

    sidereal = (24110.54841 + 8640184.812866 * t + 0.093104 * t**2 - 6.2e-6 * t**3) / 86400.
    # at Greenwich
    sid_int = sidereal.astype(np.int64)
    sidereal = sidereal - sid_int
    # longitude is measured east so add.
    sidereal = sidereal + 1.0027379093 * ut + location.lon.hour / 24.
    
    sid_int = sidereal.astype(np.int64)
    sidereal = (sidereal - sid_int) * 24.
    # if(sid < 0.) : sid = sid + 24.
    # TODO: A convertion to radians is needed if the output is not an Angle
    lst = Angle(sidereal, unit=u.hour)
    lst.wrap_at(24. * u.hour, inplace=True)

    if scalar_input:
        return np.squeeze(lst)
    return lst


def true_airmass(altit: Angle) -> npt.NDArray[float]:
    """true airmass for an altitude.
    Equivalent of getAirmass in the QPT, based on vskyutil.true_airmass
    https://github.com/gemini-hlsw/ocs/blob/12a0999bc8bb598220ddbccbdbab5aa1e601ebdd/bundle/edu.gemini.qpt.client/src/main/java/edu/gemini/qpt/core/util/ImprovedSkyCalcMethods.java#L119

    Based on a fit to Kitt Peak airmass tables, C. M. Snell & A. M. Heiser, 1968,
    PASP, 80, 336.  Valid to about airmass 12, and beyond that just returns
    secz minus 1.5, which won't be quite right.

    Takes an Angle and return the true airmass, based on a tabulation of the mean KPNO
    atmosphere given by C. M. Snell & A. M. Heiser, 1968, PASP, 80, 336.  
    They tabulated the airmass at 5 degr intervals from z = 60 to 85 degrees; I fit the data with
    a fourth order poly for (secz - airmass) as a function of (secz - 1) using 
    the IRAF curfit routine, then adjusted the zeroth order term to force (secz - airmass) to zero at
    z = 0.  The poly fit is very close to the tabulated points (largest difference is 3.2e-4) 
    and appears smooth. This 85-degree point is at secz = 11.47, so for secz > 12 just return secz

    coefs = [2.879465E-3,  3.033104E-3, 1.351167E-3, -4.716679E-5]

    Parameters
    ----------

    altit : Angle, float or numpy array
        Altitude above horizon.

    Returns : float
    """

    altit = np.asarray(altit.to_value(u.rad).data)
    scalar_input = False
    if altit.ndim == 0:
        altit = altit[None]  # Makes 1D
        scalar_input = True

    # ret = np.zeros(len(altit))
    ret = np.full(len(altit), -1.)
    ii = np.where(altit > 0.0)[0][:]
    if len(ii) != 0:
        ret[ii] = 1. / np.sin(altit[ii])  # sec z = 1/sin (altit)

    kk = np.where(np.logical_and(ret >= 0.0, ret < 12.))[0][:]
    if len(kk) != 0:
        seczmin1 = ret[kk] - 1.
        coefs = np.array([-4.716679E-5, 1.351167E-3, 3.033104E-3, 2.879465E-3, 0.])
        ret[kk] = ret[kk] - np.polyval(coefs, seczmin1)

    if scalar_input:
        return np.squeeze(ret)
    return ret


def hour_angle_to_angle(dec: AngleParam,
                        lat: AngleParam,
                        alt: AngleParam) -> Angle:
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


def xair(zd: Quantity) -> Quantity:
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


def ztwilight(alt: Angle) -> float:
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
