import numpy as np
from sky.constants import J2000
from sky.utils import current_geocent_frame, local_sidereal_time
from sky.altitude import Altitude
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time, TimeDelta
from astropy.units import u
from typing import Tuple, Union


class Sun:

    @staticmethod
    def location(time: Time):
        """
        low-precision position of the sun.

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

        n = jd - J2000  # referred to J2000
        L = 280.460 + 0.9856474 * n
        g = np.deg2rad(357.528 + 0.9856003 * n)
        lambd = np.deg2rad(L + 1.915 * np.sin(g) + 0.020 * np.sin(2. * g))
        epsilon = np.deg2rad(23.439 - 0.0000004 * n)

        x = np.cos(lambd)
        y = np.cos(epsilon) * np.sin(lambd)
        z = np.sin(epsilon) * np.sin(lambd)

        ra = np.arctan2(y, x)
        dec = np.arcsin(z)

        fr = current_geocent_frame(time)


        if scalar_input:
            ra = np.squeeze(ra)
            dec = np.squeeze(dec)
        return SkyCoord(ra, dec, frame=fr, unit='radian')

    @staticmethod
    def time_by_altitude(alt, tguess, location):   
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

        sunpos = Sun.location(tguess)
        # print "sunpos entering",sunpos
        # print "tguess.jd, longit:",tguess.jd, location.lon.hour
        tolerance = Angle(1.0e-4, unit=u.rad)

        delt = TimeDelta(0.002, format='jd')  # timestep
        # print "sidereal: ",local_sidereal_time(tguess, location)
        # print "sunpos.ra: ",sunpos.ra

        ha = local_sidereal_time(tguess, location) - sunpos.ra
        # print "ha entering",ha
        alt2, az, parang = Altitude.above(sunpos.dec, Angle(ha, unit=u.hourangle), location.lat)
        # print "alt2",alt2
        tguess = tguess + delt
        sunpos = local_sidereal_time(tguess)
        # print "sunpos with delt",sunpos
        alt3, az, parang = Altitude.above(sunpos.dec, local_sidereal_time(tguess, location) - sunpos.ra, location.lat)
        err = alt3 - alt
        # print "alt3, alt, err",alt3,alt,err
        deriv = (alt3 - alt2) / delt
        # print "deriv",deriv
        kount = np.zeros(len(tguess), dtype=int)
        kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]
        while (len(kk) != 0):
            tguess[kk] = tguess[kk] - err[kk] / deriv[kk]
            sunpos = None
            sunpos = Sun.location(tguess[kk])
            alt3[kk], az[kk], parang[kk] = Altitude.above(sunpos.dec, local_sidereal_time(tguess[kk], location) - sunpos.ra,
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
