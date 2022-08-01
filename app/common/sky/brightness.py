import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.coordinates import Angle, Distance
from astropy.units import Quantity

from app.common.minimodel import SkyBackground
from app.common.sky.constants import KZEN, EQUAT_RAD
from app.common.sky.utils import xair, ztwilight


def calculate_sky_brightness(moon_phase_angle: Angle,
                             target_moon_angdist: Distance,
                             earth_moon_dist: Distance,
                             moon_zenith_distang: Angle,
                             target_zenith_distang: Angle,
                             sun_zenith_distang: Angle,
                             verbose: bool = False) -> npt.NDArray[float]:
    """
    Calculate sky brightness based on formulas from Krisciunas & Schaefer 1991
    Uses array processing

    Parameters
    ----------
    moon_phase_angle : '~astropy.units.Quantity'
        Moon phase angles in degrees

    target_moon_angdist : '~astropy.units.Quantity'
        Angular distances between target and moon

    earth_moon_dist : '~astropy.units.Quantity'
        Distances from the Earth to the Moon

    moon_zenith_distang : '~astropy.units.Quantity'
        Moon zenith distance angles

    target_zenith_distang : '~astropy.units.Quantity'
        Target zenith distance angles

    sun_zenith_distang : '~astropy.units.Quantity'
        Sun zenith distance angles

    verbose :
        Verbose output flag

    Returns
    ---------
    skybright : array of float
        Numpy array of sky background magnitudes at target location
    """

    # Constants
    a = 2.51189
    q = 27.78151

    sun_alt = 90.0 * u.deg - sun_zenith_distang  # sun altitude
    if verbose:
        print(f'sun_alt: {sun_alt}')

    # Relative distance between the Earth and the Moon
    norm_moondist = earth_moon_dist / (60.27 * EQUAT_RAD)

    # Number of positions
    n = len(target_zenith_distang)

    # Dark sky zenith V surface brightness
    v_zen = np.ones(n) * 21.587

    # Correction for twilight
    i = np.where(sun_alt > -18.5 * u.deg)[0][:]
    if len(i) != 0:
        v_zen[i] = v_zen[i] - ztwilight(sun_alt[i])

    # zenith sky brightness
    b_zen = 0.263 * a ** (q - v_zen)

    # Sky brightness with no moon at target, scattering due to airmass
    b_sky = b_zen * xair(target_zenith_distang) * 10.0 ** (
                -0.4 * KZEN * (xair(target_zenith_distang) - 1.0))

    # Lunar sky brightness
    b_moon = np.zeros(n)
    istar = np.zeros(n)
    frho = np.zeros(n)

    # Do calculations when the Moon is above or near the horizon
    im = np.where(moon_zenith_distang < 90.8 * u.deg)[0][:]

    # Illuminance with phase angle, eq. 8
    # Correction for lunar distance.
    istar[im] = 10.0 ** (
            -0.4 * (3.84 + 0.026 * abs(moon_phase_angle[im].value) + 4.e-9 * moon_phase_angle[im].value ** 4.)
    ) / norm_moondist[im]**2

    # Rough correction for opposition effect
    # 35 per cent brighter at full, effect tapering linearly to
    #   zero at 7 degrees away from full. mentioned peripherally in
    #   Krisciunas and Scheafer, p. 1035. */
    hh = im[np.where(abs(moon_phase_angle[im]) < 7. * u.deg)[0][:]]
    if len(hh):
        istar[hh] *= (1.35 - 0.05 * abs(moon_phase_angle[hh].value))

    # Build eq. 21 piecewise, mAng is rho in skycalc and the paper
    frho[im] = 229087. * (1.06 + np.cos(target_moon_angdist[im]) ** 2.0)

    # Separation > 10 deg
    ii = im[np.where(abs(target_moon_angdist[im]) > 10.0 * u.deg)[0][:]]
    if len(ii):
        frho[ii] += 10.0 ** (6.15 - target_moon_angdist[ii].value / 40.0)

    # Separation between 0.25 and 10 deg
    jj = im[np.where(np.logical_and(abs(target_moon_angdist[im]) > 0.25 * u.deg,
                                    abs(target_moon_angdist[im]) <= 10.0 * u.deg))[0][:]]
    if len(jj):
        frho[jj] += 6.2e7 / (target_moon_angdist[jj].value ** 2.0)  # eq 19

    # If on lunar disk
    kk = im[np.where(abs(target_moon_angdist[im]) <= 0.25 * u.deg)[0][:]]
    if len(kk) != 0:
        frho[kk] += 9.9e8

    # Sky brightness from the Moon in nanoLamberts (B)
    b_moon[im] = frho[im] * istar[im] * 10 ** (-0.4 * KZEN * xair(moon_zenith_distang[im])) * (
            1.0 - 10 ** (-0.4 * KZEN * xair(target_zenith_distang[im]))
    )

    # hh = np.where(np.logical_and(cc > 0.5, cc < 0.8))[0][:]
    # if len(hh) != 0:  # very simple increase in SB if there are thin clouds
    #     b_moon[hh] = 2.0 * b_moon[hh]

    # sky brightness in Vmag/arcsec^2
    skybright = q - np.log10((b_moon + b_sky) / 0.263) / np.log10(a)

    if verbose:
        print(f'v_zen: {v_zen}')
        print(f'b_zen: {b_zen}')
        print(f'istar: {istar}')
        print(f'b_moon: {b_moon}')
        print(f'b_sky: {b_sky}')
        print(f'skybright: {skybright}')

    return skybright


def calculate_sky_brightness_qpt(moon_phase_angle: Quantity,
                                 target_moon_angdist: Quantity,
                                 moon_zenith_distang: Quantity,
                                 target_zenith_distang: Quantity,
                                 sun_zenith_distang: Quantity,
                                 verbose: bool = False) -> npt.NDArray[float]:
    """
    Calculate sky brightness based on formulas from Krisciunas & Schaefer 1991
    Bryan Miller
    November 5, 2004
    June 1, 2015 - added cc parameter while testing cloud scattering corrections

    Matt Bonnyman
    converted from IDL to Python May 23, 2018

    Parameters
    ----------
    moon_phase_angle : '~astropy.units.Quantity'
        Moon phase angle at solar midnight in degrees

    target_moon_angdist : '~astropy.units.Quantity'
        Numpy array of angular distances between target and moon

    moon_zenith_distang : '~astropy.units.Quantity'
        Numpy array of Moon zenith distance angles

    target_zenith_distang : '~astropy.units.Quantity'
        Numpy array of target zenith distance angles

    sun_zenith_distang : '~astropy.units.Quantity'
        Numpy array of Sun zenith distance angles

    verbose :
        Verbose output flag

    Returns
    ---------
    skybright : float
        Numpy array of sky background magnitudes at target location
    """

    # Constants
    a = 2.51189
    q = 27.78151

    mpaa = np.asarray(moon_phase_angle.value)
    if mpaa.ndim == 0:
        mpaa = np.ones(len(moon_zenith_distang)) * moon_phase_angle.value

    sun_alt = 90.0 * u.deg - sun_zenith_distang  # sun altitude
    if verbose:
        print(f'sun_alt: {sun_alt}')

    # Dark sky zenith V surface brightness
    v_zen = np.ones(len(target_zenith_distang)) * 21.587
    # correction for twilight
    ii = np.where(sun_alt > -18.5 * u.deg)[0][:]
    v_zen[ii] = v_zen[ii] - ztwilight(sun_alt[ii])

    # zenith sky brightness
    b_zen = 0.263 * a**(q - v_zen)

    # Sky brightness with no moon at target, scattering due to airmass
    b_sky = b_zen * xair(target_zenith_distang) * 10.0 ** (
            -0.4 * KZEN * (xair(target_zenith_distang) - 1.0)
    )

    # Lunar sky brightness
    n = len(b_sky)
    b_moon = np.zeros(n)

    istar = 10.0 ** (-0.4 * (3.84 + 0.026 * abs(mpaa) + 4.e-9 * mpaa**4.))

    ii = np.where(moon_zenith_distang < 90.8 * u.deg)[0][:]

    jj = ii[np.where(target_moon_angdist[ii] >= 10.0 * u.deg)[0][:]]
    if len(jj):
        fpjj = (1.06 + np.cos(target_moon_angdist[jj]) ** 2) * 10.0 ** 5.36 + 10.0 ** (
                6.15 - target_moon_angdist[jj].value / 40.0
        )
        b_moon[jj] = fpjj * istar[jj] * 10 ** (-0.4 * KZEN * xair(moon_zenith_distang[jj])) * (
                1.0 - 10 ** (-0.4 * KZEN * xair(target_zenith_distang[jj]))
        )

    kk = np.where(ii != jj)[0][:]
    if len(kk):
        # There is a bug in the following line from the original code, used by QPT
        fpkk = 6.2e7 / (target_moon_angdist[kk].value ** 2)
        # fpkk = (1.06 + np.cos(mdist[kk])**2) * 10.0**5.36 + 6.2e7 / (mdist[kk].value**2)
        b_moon[kk] = fpkk * istar[kk] * 10 ** (-0.4 * KZEN * xair(moon_zenith_distang[kk])) * (
                1.0 - 10 ** (-0.4 * KZEN * xair(target_zenith_distang[kk]))
        )

    # hh = np.where(np.logical_and(cc > 0.5, cc < 0.8))[0][:]
    # if len(hh) != 0:  # very simple increase in SB if there are thin clouds
    #     b_moon[hh] = 2.0 * b_moon[hh]

    skybright = q - np.log10((b_moon + b_sky) / 0.263) / np.log10(a)  # sky brightness in Vmag/arcsec^2

    if verbose:
        print(f'v_zen: {v_zen}')
        print(f'b_zen: {b_zen}')
        print(f'istar: {istar}')
        print(f'b_moon: {b_moon}')
        print(f'b_sky: {b_sky}')
        print(f'skybright: {skybright}')

    return skybright


def convert_to_sky_background(sb: npt.NDArray[float]) -> npt.NDArray[SkyBackground]:
    """
    Convert visible sky background magnitudes to decimal conditions.

        Conversion scheme:
            1.0 |          vsb <= 19.61
            0.8 | 19.61 < vsb <= 20.78
            0.5 | 20.78 < vsb <= 21.37
            0.2 | 21.37 < vsb


    Input
    -------
    sb :  np.ndarray of floats
        TargetInfo object with time dependent vsb quantities

    Return
    -------
    cond : np.ndarray of SkyBackground
        sky background condition values
    """

    cond = np.full([len(sb)], SkyBackground.SBANY)

    ii = np.where(np.logical_and(sb > 19.61, sb <= 20.78))[0][:]
    cond[ii] = SkyBackground.SB80

    ii = np.where(np.logical_and(sb > 20.78, sb <= 21.37))[0][:]
    cond[ii] = SkyBackground.SB50

    ii = np.where(sb > 21.37)[0][:]
    cond[ii] = SkyBackground.SB20

    return cond
