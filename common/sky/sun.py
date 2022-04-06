import numpy as np
from common.sky.constants import J2000
from common.sky.utils import current_geocent_frame, local_sidereal_time
from common.sky.altitude import Altitude
from astropy.coordinates import Angle, SkyCoord, EarthLocation
from astropy.time import Time, TimeDelta
import astropy.units as u

class Sun:

    @staticmethod
    def at(time: Time) -> SkyCoord:
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

        frame = current_geocent_frame(time)

        if scalar_input:
            ra = np.squeeze(ra)
            dec = np.squeeze(dec)
        return SkyCoord(ra, dec, frame=frame, unit='radian')

    @staticmethod
    def time_by_altitude(alt: Angle,
                         time_guess: Time,
                         location: EarthLocation,
                         timestep: float = 0.002) -> Time:
        """
        Time at which the sun crosses a particular altitude alt, which is an Angle,
        for an EarthLocation location.

        This of course happens twice a day (or not at all);
        time_guess is a Time approximating the answer.
        The usual use case will be to compute roughly when sunset or twilight occurs,
        and hand the result to this routine to get a more exact answer.

        This uses the low-precision sun location, which is typically good to 0.01 degree.
        That's plenty good enough for computing rise, set, and twilight times.

        Parameters:

        alt : Angle, single or array. If array, then must be the same length as time_guess
        Desired altitude.
        time_guess : Time, single or array
        Starting time for iteration.  This must be fairly
        close so that the iteration coverges on the correct
        phenomenon (e.g., rise time, not set time).
        location : EarthLocation

        Returns: Time if convergent
                None if non-convergent
        """
        time_guess = Time(np.asarray(time_guess.jd), format='jd')
        alt = Angle(np.asarray(alt.to_value(u.rad)), unit=u.rad)
        scalar_input = False
        if time_guess.ndim == 0 and alt.ndim == 0:
            scalar_input = True
        if time_guess.ndim == 0:
            time_guess = time_guess[None]  # Makes 1D
        if alt.ndim == 0:
            alt = alt[None]

        if len(time_guess) == 1 and len(alt) > 1:
            time_guess = Time(time_guess.jd * np.ones(len(alt)), format='jd')
        elif len(time_guess) > 1 and len(alt) == 1:
            alt = alt * np.ones(len(time_guess))
        elif len(time_guess) != len(alt):
            raise ValueError('Error: alt and time_guess have incompatible lengths')

        sun_pos = Sun.at(time_guess)
        tolerance = Angle(1.0e-4, unit=u.rad)

        delta = TimeDelta(timestep, format='jd')
   
        ha = local_sidereal_time(time_guess, location) - sun_pos.ra
        alt2, az, parang = Altitude.above(sun_pos.dec, Angle(ha, unit=u.hourangle), location.lat)

        time_guess = time_guess + delta
        sun_pos = Sun.at(time_guess)

        alt3, az, parang = Altitude.above(sun_pos.dec,
                                          local_sidereal_time(time_guess, location) - sun_pos.ra, 
                                          location.lat)
        err = alt3 - alt
        deriv = (alt3 - alt2) / delta

        kount = np.zeros(len(time_guess), dtype=int)
        kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]
        while (len(kk) != 0):
            time_guess[kk] = time_guess[kk] - err[kk] / deriv[kk]
            sun_pos = None
            sun_pos = Sun.at(time_guess[kk])
            alt3[kk], az[kk], parang[kk] = Altitude.above(sun_pos.dec,
                                                          local_sidereal_time(time_guess[kk], location) - sun_pos.ra,
                                                          location.lat)
            err[kk] = alt3[kk] - alt[kk]
            kount[kk] = kount[kk] + 1
            ii = np.where(kount >= 9)[0][:]
            if len(ii) != 0:
                print("Sunrise, set, or twilight calculation not converging!\n")
                return None
            kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]

        if scalar_input:
            time_guess = np.squeeze(time_guess)
        return Time(time_guess, format='iso')