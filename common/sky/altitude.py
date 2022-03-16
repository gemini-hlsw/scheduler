import numpy as np
import numpy.typing as npt
import astropy.units as u
from astropy.coordinates import Angle, Longitude
from typing import Tuple, Union

class Altitude:

    def above(dec: Union[Angle, float, npt.NDArray[float]],
              ha: Union[Angle, float, npt.NDArray[float]],
              lat: Union[Angle, float, npt.NDArray[float]]) -> Tuple[Angle, Angle, Angle]:
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
