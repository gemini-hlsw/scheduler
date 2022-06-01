from typing import Tuple, Union, TypeVar

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import Angle, Longitude

T = TypeVar('T')
U = TypeVar('U')
ScalarOrArray = Union[T, U, npt.NDArray[U]]
AngleParam = ScalarOrArray[Angle, float]


class Altitude:
    def __init__(self):
        raise NotImplementedError('Use static method Altitude.above.')

    @staticmethod
    def above(dec: AngleParam,
              ha: AngleParam,
              lat: AngleParam) -> Tuple[Angle, Angle, Angle]:
        """
        Compute altitude above horizon, azimuth, and parallactic angle.

        The parallactic angle is the position angle of the arc between the
        object and the zenith, i.e. the position angle that points 'straight up'
        when you're looking at an object.  It is important for atmospheric
        refraction and dispersion compensation, which Filippenko discusses
        ( 1982PASP...94..715F ).  Does not take small effects into
        account (polar motion, nutation, whatever) so is lighter weight than
        the astropy equivalents.

        Filippenko's expression for the parallactic angle leaves it to the user
        to select the correct root of an inverse trig function; this is handled
        automatically here by fully solving the astronomical triangle.

        The astropy altaz transformation depends on the 3 Mbyte download from USNO
        to find the lst, so here is a stripped down version.
        # Arguments are all assumed to be Angles so they don't need to be converted to radians;
        the dec is assumed to be in equinox of date to avoid
        # We get the parallactic angle almost for since we have ha, colat, and altitude.

        Parameters
        ----------

        dec : Angle, float or numpy array
            Declination
        ha : Angle, float or numpy array
            Hour angle (spherical astronomy) of the position, positive westward
        lat : Angle
            Latitude of site.

        Returns
        -------
        tuple of (altitude, azimuth, parallactic), all of which are Angles.
        """

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
            raise ValueError('Error: dec and ha have incompatible lengths')

        cos_dec = np.cos(dec)
        sin_dec = np.sin(dec)
        cos_ha = np.cos(ha)
        sin_ha = np.sin(ha)
        cos_lat = np.cos(lat)
        sinlat = np.sin(lat)

        # Compute the altitude
        altitude = Angle(np.arcsin(cos_dec * cos_ha * cos_lat + sin_dec * sinlat), unit=u.radian)

        # Compute the azimuth
        y = sin_dec * cos_lat - cos_dec * cos_ha * sinlat  # due north component
        z = -1. * cos_dec * sin_ha  # due east component
        azimuth = Longitude(np.arctan2(z, y), unit=u.radian)

        # Now solve the spherical trig to give parallactic angle
        parang = np.zeros(len(dec))
        ii = np.where(cos_dec != 0.0)[0][:]
        if len(ii) != 0:
            # spherical law of sines
            sinp = -1. * np.sin(azimuth[ii]) * cos_lat / cos_dec[ii]
            # spherical law of cosines, also expressed in already computed variables.
            cosp = -1. * np.cos(azimuth[ii]) * cos_ha[ii] - np.sin(azimuth[ii]) * sin_ha[ii] * sinlat
            parang[ii] = np.arctan2(sinp, cosp)

        jj = np.where(cos_dec == 0.0)[0][:]  # you're on the pole
        if len(jj) != 0:
            parang[jj] = np.pi

        if scalar_input:
            altitude = np.squeeze(altitude)
            azimuth = np.squeeze(azimuth)
            parang = np.squeeze(parang)
        return altitude, azimuth, Angle(parang, unit=u.rad)
