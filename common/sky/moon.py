from typing import NoReturn, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, Distance, GeocentricTrueEcliptic, Angle
from astropy.time import Time
from astropy.time import TimeDelta

from common.sky.altitude import Altitude
from common.sky.constants import J2000, EQUAT_RAD
from common.sky.utils import local_sidereal_time, current_geocent_frame, geocentric_coors, hour_angle_to_angle


class Moon:
    def __init__(self):
        # TODO: skip constructor altogether and just use the at method?
        self.time = None
        self.time_jd = None
        self.time_ttjd = None  # important to use correct time argument for this!!
        self.scalar_input = False
        self.pie = None
        self.lambd = None
        self.beta = None

    def at(self, time: Time):
        self.time = time
        self.time_jd = np.asarray(time.jd)
        self.time_ttjd = np.asarray(time.tt.jd)  # important to use correct time argument for this!!
        self.scalar_input = False
        if self.time_ttjd.ndim == 0:
            self.time_ttjd = self.time_ttjd[None]  # Makes 1D
            self.scalar_input = True

        return self

    def _low_precision_calculations(self) -> NoReturn:
        """
        Compute low precision values for the moon location method calculations.
        """
        t = (self.time_jd - J2000) / 36525.  # jul cent. since J2000.0

        lambd = (218.32 + 481267.883 * t
                 + 6.29 * np.sin(np.deg2rad(134.9 + 477198.85 * t))
                 - 1.27 * np.sin(np.deg2rad(259.2 - 413335.38 * t))
                 + 0.66 * np.sin(np.deg2rad(235.7 + 890534.23 * t))
                 + 0.21 * np.sin(np.deg2rad(269.9 + 954397.70 * t))
                 - 0.19 * np.sin(np.deg2rad(357.5 + 35999.05 * t))
                 - 0.11 * np.sin(np.deg2rad(186.6 + 966404.05 * t)))
        self.lambd = np.deg2rad(lambd)

        beta = (5.13 * np.sin(np.deg2rad(93.3 + 483202.03 * t))
                + 0.28 * np.sin(np.deg2rad(228.2 + 960400.87 * t))
                - 0.28 * np.sin(np.deg2rad(318.3 + 6003.18 * t))
                - 0.17 * np.sin(np.deg2rad(217.6 - 407332.20 * t)))
        self.beta = np.deg2rad(beta)

        pie = (0.9508 + 0.0518 * np.cos(np.deg2rad(134.9 + 477198.85 * t))
               + 0.0095 * np.cos(np.deg2rad(259.2 - 413335.38 * t))
               + 0.0078 * np.cos(np.deg2rad(235.7 + 890534.23 * t))
               + 0.0028 * np.cos(np.deg2rad(269.9 + 954397.70 * t)))
        self.pie = np.deg2rad(pie)

    def _high_precision_calculations(self) -> NoReturn:
        """
        Compute accurate precession values for the moon location method calculations.
        """
        t = (self.time_ttjd - 2415020.) / 36525.  # this based around 1900 ... */
        t_2 = t * t
        t_3 = t_2 * t
        lpr = 270.434164 + 481267.8831 * t - 0.001133 * t_2 + 0.0000019 * t_3
        m = 358.475833 + 35999.0498 * t - 0.000150 * t_2 - 0.0000033 * t_3
        mpr = 296.104608 + 477198.8491 * t + 0.009192 * t_2 + 0.0000144 * t_3
        d = 350.737486 + 445267.1142 * t - 0.001436 * t_2 + 0.0000019 * t_3
        f = 11.250889 + 483202.0251 * t - 0.003211 * t_2 - 0.0000003 * t_3
        om = 259.183275 - 1934.1420 * t + 0.002078 * t_2 + 0.0000022 * t_3

        lpr = lpr % 360.
        mpr = mpr % 360.
        m = m % 360.
        d = d % 360.
        f = f % 360.
        om = om % 360.

        sinx = np.sin(np.deg2rad(51.2 + 20.2 * t))
        lpr = lpr + 0.000233 * sinx
        m = m - 0.001778 * sinx
        mpr = mpr + 0.000817 * sinx
        d = d + 0.002011 * sinx

        sinx = 0.003964 * np.sin(np.deg2rad(346.560 + 132.870 * t - 0.0091731 * t_2))

        lpr = lpr + sinx
        mpr = mpr + sinx
        d = d + sinx
        f = f + sinx

        sinx = np.sin(np.deg2rad(om))
        lpr = lpr + 0.001964 * sinx
        mpr = mpr + 0.002541 * sinx
        d = d + 0.001964 * sinx
        f = f - 0.024691 * sinx
        f = f - 0.004328 * np.sin(np.deg2rad(om + 275.05 - 2.30 * t))

        e = 1 - 0.002495 * t - 0.00000752 * t_2

        m = np.deg2rad(m)  # these will all be arguments ... */
        mpr = np.deg2rad(mpr)
        d = np.deg2rad(d)
        f = np.deg2rad(f)

        lambd = (lpr + 6.288750 * np.sin(mpr)
                 + 1.274018 * np.sin(2 * d - mpr)
                 + 0.658309 * np.sin(2 * d)
                 + 0.213616 * np.sin(2 * mpr)
                 - e * 0.185596 * np.sin(m)
                 - 0.114336 * np.sin(2 * f)
                 + 0.058793 * np.sin(2 * d - 2 * mpr)
                 + e * 0.057212 * np.sin(2 * d - m - mpr)
                 + 0.053320 * np.sin(2 * d + mpr)
                 + e * 0.045874 * np.sin(2 * d - m)
                 + e * 0.041024 * np.sin(mpr - m)
                 - 0.034718 * np.sin(d)
                 - e * 0.030465 * np.sin(m + mpr)
                 + 0.015326 * np.sin(2 * d - 2 * f)
                 - 0.012528 * np.sin(2 * f + mpr)
                 - 0.010980 * np.sin(2 * f - mpr)
                 + 0.010674 * np.sin(4 * d - mpr)
                 + 0.010034 * np.sin(3 * mpr)
                 + 0.008548 * np.sin(4 * d - 2 * mpr)
                 - e * 0.007910 * np.sin(m - mpr + 2 * d)
                 - e * 0.006783 * np.sin(2 * d + m)
                 + 0.005162 * np.sin(mpr - d))

        lambd = (lambd + e * 0.005000 * np.sin(m + d)
                 + e * 0.004049 * np.sin(mpr - m + 2 * d)
                 + 0.003996 * np.sin(2 * mpr + 2 * d)
                 + 0.003862 * np.sin(4 * d)
                 + 0.003665 * np.sin(2 * d - 3 * mpr)
                 + e * 0.002695 * np.sin(2 * mpr - m)
                 + 0.002602 * np.sin(mpr - 2 * f - 2 * d)
                 + e * 0.002396 * np.sin(2 * d - m - 2 * mpr)
                 - 0.002349 * np.sin(mpr + d)
                 + e * e * 0.002249 * np.sin(2 * d - 2 * m)
                 - e * 0.002125 * np.sin(2 * mpr + m)
                 - e * e * 0.002079 * np.sin(2 * m)
                 + e * e * 0.002059 * np.sin(2 * d - mpr - 2 * m)
                 - 0.001773 * np.sin(mpr + 2 * d - 2 * f)
                 - 0.001595 * np.sin(2 * f + 2 * d)
                 + e * 0.001220 * np.sin(4 * d - m - mpr)
                 - 0.001110 * np.sin(2 * mpr + 2 * f)
                 + 0.000892 * np.sin(mpr - 3 * d)
                 - e * 0.000811 * np.sin(m + mpr + 2 * d)
                 + e * 0.000761 * np.sin(4 * d - m - 2 * mpr)
                 + e * e * 0.000717 * np.sin(mpr - 2 * m)
                 + e * e * 0.000704 * np.sin(mpr - 2 * m - 2 * d)
                 + e * 0.000693 * np.sin(m - 2 * mpr + 2 * d)
                 + e * 0.000598 * np.sin(2 * d - m - 2 * f)
                 + 0.000550 * np.sin(mpr + 4 * d)
                 + 0.000538 * np.sin(4 * mpr)
                 + e * 0.000521 * np.sin(4 * d - m)
                 + 0.000486 * np.sin(2 * mpr - d))

        b = (5.128189 * np.sin(f)
             + 0.280606 * np.sin(mpr + f)
             + 0.277693 * np.sin(mpr - f)
             + 0.173238 * np.sin(2 * d - f)
             + 0.055413 * np.sin(2 * d + f - mpr)
             + 0.046272 * np.sin(2 * d - f - mpr)
             + 0.032573 * np.sin(2 * d + f)
             + 0.017198 * np.sin(2 * mpr + f)
             + 0.009267 * np.sin(2 * d + mpr - f)
             + 0.008823 * np.sin(2 * mpr - f)
             + e * 0.008247 * np.sin(2 * d - m - f)
             + 0.004323 * np.sin(2 * d - f - 2 * mpr)
             + 0.004200 * np.sin(2 * d + f + mpr)
             + e * 0.003372 * np.sin(f - m - 2 * d)
             + 0.002472 * np.sin(2 * d + f - m - mpr)
             + e * 0.002222 * np.sin(2 * d + f - m)
             + e * 0.002072 * np.sin(2 * d - f - m - mpr)
             + e * 0.001877 * np.sin(f - m + mpr)
             + 0.001828 * np.sin(4 * d - f - mpr)
             - e * 0.001803 * np.sin(f + m)
             - 0.001750 * np.sin(3 * f)
             + e * 0.001570 * np.sin(mpr - m - f)
             - 0.001487 * np.sin(f + d)
             - e * 0.001481 * np.sin(f + m + mpr)
             + e * 0.001417 * np.sin(f - m - mpr)
             + e * 0.001350 * np.sin(f - m)
             + 0.001330 * np.sin(f - d)
             + 0.001106 * np.sin(f + 3 * mpr)
             + 0.001020 * np.sin(4 * d - f)
             + 0.000833 * np.sin(f + 4 * d - mpr))

        b = (b + 0.000781 * np.sin(mpr - 3 * f)
             + 0.000670 * np.sin(f + 4 * d - 2 * mpr)
             + 0.000606 * np.sin(2 * d - 3 * f)
             + 0.000597 * np.sin(2 * d + 2 * mpr - f)
             + e * 0.000492 * np.sin(2 * d + mpr - m - f)
             + 0.000450 * np.sin(2 * mpr - f - 2 * d)
             + 0.000439 * np.sin(3 * mpr - f)
             + 0.000423 * np.sin(f + 2 * d + 2 * mpr)
             + 0.000422 * np.sin(2 * d - f - 3 * mpr)
             - e * 0.000367 * np.sin(m + f + 2 * d - mpr)
             - e * 0.000353 * np.sin(m + f + 2 * d)
             + 0.000331 * np.sin(f + 4 * d)
             + e * 0.000317 * np.sin(2 * d + f - m + mpr)
             + e * e * 0.000306 * np.sin(2 * d - 2 * m - f)
             - 0.000283 * np.sin(mpr + 3 * f))

        om1 = 0.0004664 * np.cos(np.deg2rad(om))
        om2 = 0.0000754 * np.cos(np.deg2rad(om + 275.05 - 2.30 * t))

        beta = b * (1. - om1 - om2)

        self.pie = (0.950724 + 0.051818 * np.cos(mpr)
                    + 0.009531 * np.cos(2 * d - mpr)
                    + 0.007843 * np.cos(2 * d)
                    + 0.002824 * np.cos(2 * mpr)
                    + 0.000857 * np.cos(2 * d + mpr)
                    + e * 0.000533 * np.cos(2 * d - m)
                    + e * 0.000401 * np.cos(2 * d - m - mpr)
                    + e * 0.000320 * np.cos(mpr - m)
                    - 0.000271 * np.cos(d)
                    - e * 0.000264 * np.cos(m + mpr)
                    - 0.000198 * np.cos(2 * f - mpr)
                    + 0.000173 * np.cos(3 * mpr)
                    + 0.000167 * np.cos(4 * d - mpr)
                    - e * 0.000111 * np.cos(m)
                    + 0.000103 * np.cos(4 * d - 2 * mpr)
                    - 0.000084 * np.cos(2 * mpr - 2 * d)
                    - e * 0.000083 * np.cos(2 * d + m)
                    + 0.000079 * np.cos(2 * d + 2 * mpr)
                    + 0.000072 * np.cos(4 * d)
                    + e * 0.000064 * np.cos(2 * d - m + mpr)
                    - e * 0.000063 * np.cos(2 * d + m - mpr)
                    + e * 0.000041 * np.cos(m + d)
                    + e * 0.000035 * np.cos(2 * mpr - m)
                    - 0.000033 * np.cos(3 * mpr - 2 * d)
                    - 0.000030 * np.cos(mpr + d)
                    - 0.000029 * np.cos(2 * f - 2 * d)
                    - e * 0.000029 * np.cos(2 * mpr + m)
                    + e * e * 0.000026 * np.cos(2 * d - 2 * m)
                    - 0.000023 * np.cos(2 * f - 2 * d + mpr)
                    + e * 0.000019 * np.cos(4 * d - m - mpr))

        self.beta = Angle(np.deg2rad(beta), unit=u.rad)
        self.lambd = Angle(np.deg2rad(lambd), unit=u.rad)

    def low_precision_location(self, obs: EarthLocation) -> Tuple[SkyCoord, float]:
        """
        This is the same as the high precision method, but with a different set of coefficients.  
        The difference is small. Good to about 0.1 deg, from the 1992 Astronomical Almanac, p. D46.
        Note that input time is a float.
        """
        self._low_precision_calculations()

        # Terrestrial time with julian day
        sid = local_sidereal_time(self.time, obs)
        lat = obs.lat
        distance = 1. / np.sin(self.pie)

        ell = np.cos(self.beta) * np.cos(self.lambd)
        m = 0.9175 * np.cos(self.beta) * np.sin(self.lambd) - 0.3978 * np.sin(self.beta)
        n = 0.3978 * np.cos(self.beta) * np.sin(self.lambd) + 0.9175 * np.sin(self.beta)

        x = ell * distance
        y = m * distance
        z = n * distance

        x = x - np.cos(lat) * np.cos(sid)
        y = y - np.cos(lat) * np.sin(sid)
        z = z - np.sin(lat)

        topo_dist = np.sqrt(x * x + y * y + z * z)

        ell = x / topo_dist
        m = y / topo_dist
        n = z / topo_dist

        alpha = np.arctan2(m, ell)
        delta = np.arcsin(n)
        distancemultiplier = Distance(EQUAT_RAD, unit=u.m)

        fr = current_geocent_frame(self.time)
        return SkyCoord(alpha, delta, topo_dist * distancemultiplier, frame=fr), topo_dist

    def accurate_location(self, obs: EarthLocation) -> Tuple[SkyCoord, float]:
        """  
        Compute topocentric location and distance of moon to better accuracy.

        This is good to about 0.01 degrees

        Parameters
        ----------
        obs : EarthLocation
            location on earth.

        Returns
        -------
        tuple of a SkyCoord and a distance.

        """
        self._high_precision_calculations()
        dist = Distance(1. / np.sin(np.deg2rad(self.pie)) * EQUAT_RAD)

        # place these in a skycoord in ecliptic coords of date.  Handle distance
        # separately since it does not transform properly for some reason.

        # eq = 'J{:7.2f}'.format(2000. + (time_ttjd[0] - _Constants.J2000) / 365.25)
        equinox = f'J{2000. + (self.time_ttjd[0] - J2000) / 365.25:7.2f}'
        frame = GeocentricTrueEcliptic(equinox=equinox)
        inecl = SkyCoord(lon=Angle(self.lambd, unit=u.rad), lat=Angle(self.beta, unit=u.rad), frame=frame)

        # Transform into geocentric equatorial.
        geocen = inecl.transform_to(current_geocent_frame(self.time))

        # Do the topo correction yourself. First form xyz coords in equatorial syst of date
        x = dist * np.cos(geocen.ra) * np.cos(geocen.dec)
        y = dist * np.sin(geocen.ra) * np.cos(geocen.dec)
        z = dist * np.sin(geocen.dec)

        # Now compute geocentric location of the observatory in a frame aligned with the
        # equatorial system of date, which one can do simply by replacing the west longitude
        # with the sidereal time

        # Exact match with thorskyutil/skycalc with the line below
        xobs, yobs, zobs = geocentric_coors(local_sidereal_time(self.time, obs), obs.lat, obs.height)

        # recenter moon's cartesian coordinates on position of obs
        x = x - xobs
        y = y - yobs
        z = z - zobs

        # form the toposcentric ra and dec and bundle them into a skycoord of epoch of date.
        topo_dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        raout = np.arctan2(y, x)
        decout = np.arcsin(z / topo_dist)

        if self.scalar_input:
            raout = np.squeeze(raout)
            decout = np.squeeze(decout)
            topo_dist = np.squeeze(topo_dist)

        return SkyCoord(raout, decout, unit=u.rad, frame=current_geocent_frame(self.time)), topo_dist

    @staticmethod
    def time_by_altitude(alt: Angle,
                         time_guess: Time,
                         location: EarthLocation) -> Time:
        """
        Time at which moon passes a given altitude.

        This really does have to be iterated since the moon moves fairly
        quickly.

        Parameters
        ----------
        alt : Angle, single or array.  If array, then must be the same length as time_guess
        desired altitude.
        time_guess : Time, single or array
        initial guess; this needs to be fairly close.
        location : EarthLocation

        Returns
        -------
        a Time, or None if non-convergent.
        """

        # time_guess is a Time, location is an EarthLocation

        time_guess = Time(np.asarray(time_guess.jd), format='jd')
        scalar_input = False
        if time_guess.ndim == 0:
            time_guess = time_guess[None]  # Makes 1D
            scalar_input = True
        alt = Angle(np.asarray(alt.to_value(u.rad)), unit=u.rad)
        if alt.ndim == 0:
            alt = alt[None]

        if len(time_guess) != len(alt):
            raise ValueError('Error: alt and guess must be the same length')

        moon = Moon()
        moon_pos, _ = moon.at(time_guess).accurate_location(location)
        tolerance = Angle(1.0e-4, unit=u.rad)

        delta = TimeDelta(0.002, format='jd')  # timestep

        ha = local_sidereal_time(time_guess, location) - moon_pos.ra

        alt2, az, parang = Altitude.above(moon_pos.dec, Angle(ha, unit=u.hourangle), location.lat)

        time_guess += delta
        moon_pos, _ = moon.at(time_guess).accurate_location(location)

        alt3, az, parang = Altitude.above(moon_pos.dec, local_sidereal_time(time_guess, location) - moon_pos.ra,
                                          location.lat)
        err = alt3 - alt

        deriv = (alt3 - alt2) / delta

        kount = np.zeros(len(time_guess), dtype=int)
        kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]
        while len(kk) != 0:

            time_guess[kk] = time_guess[kk] - err[kk] / deriv[kk]

            moon_pos, _ = moon.at(time_guess[kk]).accurate_location(location)
            alt3[kk], az[kk], parang[kk] = Altitude.above(moon_pos.dec,
                                                          local_sidereal_time(time_guess[kk], location) - moon_pos.ra,
                                                          location.lat)
            err[kk] = alt3[kk] - alt[kk]
            ii = np.where(kount >= 9)[0][:]
            if len(ii) != 0:
                raise ValueError("Moonrise or set calculation not converging!\n")

            kk = np.where(np.logical_and(abs(err) > tolerance, kount < 10))[0][:]

        if scalar_input:
            time_guess = np.squeeze(time_guess)
        return Time(time_guess, format='iso')

    def rise_and_set(self,
                     location: EarthLocation,
                     midnight: Time,
                     set_alt: Angle,
                     rise_alt: Angle) -> Tuple[Time, Tuple[Angle, Time, EarthLocation]]:
        """
        Return times of moon rise and set.
        """
        moon_at_midnight, _ = self.at(midnight).low_precision_location(location)
        lst_midnight = local_sidereal_time(midnight, location)
        ha_moon_at_midnight = lst_midnight - moon_at_midnight.ra
        ha_moon_at_midnight.wrap_at(12. * u.hour, inplace=True)

        ha_moon_set = hour_angle_to_angle(moon_at_midnight.dec, location.lat, set_alt)  # corresponding hr angles
        diff_moon_set = ha_moon_set - ha_moon_at_midnight  # how far from setting point at midnight

        # find nearest setting point
        ii = np.where(diff_moon_set.hour >= 12.)[0][:]
        if len(ii) != 0:
            diff_moon_set[ii] = diff_moon_set[ii] - Angle(24. * u.hour)

        jj = np.where(diff_moon_set.hour < -12.)[0][:]
        if len(jj) != 0:
            diff_moon_set[jj] = diff_moon_set[jj] + Angle(24. * u.hour)

        timedelta_moon_set = TimeDelta(diff_moon_set.hour / 24., format='jd')
        times_moon_set = midnight + timedelta_moon_set
        times_moon_set = (set_alt, times_moon_set, location)

        ha_moonrise = -1. * hour_angle_to_angle(moon_at_midnight.dec, location.lat, rise_alt)
        diff_moonrise = ha_moonrise - ha_moon_at_midnight  # how far from riseting point at midnight

        # find nearest riseting point
        ii = np.where(diff_moonrise.hour >= 12.)[0][:]
        if len(ii) != 0:
            diff_moonrise[ii] = diff_moonrise[ii] - Angle(24. * u.hour)
        jj = np.where(diff_moonrise.hour < -12.)[0][:]
        if len(jj) != 0:
            diff_moonrise[jj] = diff_moonrise[jj] + Angle(24. * u.hour)

        timedelta_moonrise = TimeDelta(diff_moonrise.hour / 24., format='jd')
        times_moonrise = midnight + timedelta_moonrise
        times_moonrise = Moon.time_by_altitude(rise_alt, times_moonrise, location)

        return times_moonrise, times_moon_set
