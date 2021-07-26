#!/usr/bin/env python3

# Routines related to calculating sky brightness

import re
import numpy as np
from astropy import units as u
from vskyutil import ztwilight, xair


def convertcond(iq, cc, bg, wv, verbose=False):
    """
    Convert actual weather conditions
    to decimal values in range [0,1].
    Conditions 'Any' or 'null' are assigned 1.
    Percentages converted to decimals.

    Accepts str or numpy.array of str.

    Parameters
    ----------
    iq : string or np.ndarray
        Image quality

    cc : string or np.ndarray
        Cloud condition

    bg : string or np.ndarray
        Sky background

    wv : string or np.ndarray
        Water vapor

    Returns
    ---------
    iq : float or np.ndarray
        Image quality

    cc : float or np.ndarray
        Cloud condition

    bg : float or np.ndarray
        Sky background

    wv : float or np.ndarray
        Water vapor

    verbose : boolean
        Verbose output?
    """

    # verbose = False

    if verbose:
        print(' Inputs conds...')
        print(' iq', iq)
        print(' cc', cc)
        print(' bg', bg)
        print(' wv', wv)

    errormessage = 'Must be type str, float, or np.ndarray'

    if isinstance(iq, np.ndarray):
        iq = np.array(list(map(conviq, iq)))
    elif isinstance(iq, str) or isinstance(iq, float):
        iq = conviq(iq)
    else:
        raise ValueError(errormessage)

    if isinstance(cc, np.ndarray):
        cc = np.array(list(map(convcc, cc)))
    elif isinstance(cc, str) or isinstance(cc, float):
        cc = convcc(cc)
    else:
        raise ValueError(errormessage)

    if isinstance(bg, np.ndarray):
        bg = np.array(list(map(convbg, bg)))
    elif isinstance(bg, str) or isinstance(bg, float):
        bg = convbg(bg)
    else:
        raise ValueError(errormessage)

    if isinstance(wv, np.ndarray):
        wv = np.array(list(map(convwv, wv)))
    elif isinstance(wv, str) or isinstance(wv, float):
        wv = convwv(wv)
    else:
        raise ValueError(errormessage)

    if verbose:
        print(' Converted conds...')
        print(' iq', iq)
        print(' cc', cc)
        print(' bg', bg)
        print(' wv', wv)

    return iq, cc, bg, wv


def conviq(string):
    """
    Convert image quality percentile string to decimal value.
    """
    if np.logical_or('any' in string.lower(), 'null' in string.lower()):
        iq = 1.
    else:
        iq = float(re.findall(r'[\d\.\d]+', string)[0])/100.
        if 0.0 < iq <= 0.2:
            iq = 0.2
#        elif 0.2 < iq <= 0.5:
#            iq = 0.5
        elif 0.2 < iq <= 0.7:
            iq = 0.7
        elif 0.7 < iq <= 0.85:
            iq = 0.85
        else:
            iq = 1.
    return iq


def convcc(string):
    """
    Convert cloud condition percentile string to decimal value.
    """
    if np.logical_or('any' in string.lower(), 'null' in string.lower()):
        cc = 1.
    else:
        cc = float(re.findall(r'[\d\.\d]+',string)[0])/100.
        if 0.0 < cc <= 0.5:
#            cc = 0.2
#        elif 0.2 < cc <= 0.5:
            cc = 0.5
        elif 0.5 < cc <= 0.7:
            cc = 0.7
        elif 0.5 < cc <= 0.80:
            cc = 0.80
        else:
            cc = 1.
    return cc


def convbg(string):
    """
    Convert sky background percentile string to decimal value.
    """
    if np.logical_or('any' in string.lower(), 'null' in string.lower()):
        bg = 1.
    else:
        bg = float(re.findall(r'[\d\.\d]+',string)[0])/100.
        if 0.0 < bg <= 0.2:
            bg = 0.2
        elif 0.2 < bg <= 0.5:
            bg = 0.5
#        elif 0.5 < bg <= 0.7:
#            bg = 0.7
        elif 0.5 < bg <= 0.80:
            bg = 0.80
        else:
            bg = 1.
    return bg


def convwv(string):
    """
    Convert water vapour percentile string to decimal value.
    """
    if np.logical_or('any' in string.lower(), 'null' in string.lower()):
        wv = 1.
    else:
        wv = float(re.findall(r'[\d\.\d]+', string)[0]) / 100
        if 0.0 < wv <= 0.2:
            wv = 0.2
        elif 0.2 < wv <= 0.5:
            wv = 0.5
#        elif 0.5 < wv <= 0.7:
#            wv = 0.7
        elif 0.5 < wv <= 0.80:
            wv = 0.80
        else:
            wv = 1.
    return wv


def sb_to_cond(sb):
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
    cond : np.ndarray of floats
        sky background condition values
    """

    cond = np.ones(len(sb), dtype=float)
    # cond = np.empty(len(sb), dtype=float)
    # ii = np.where(sb < 19.61)[0][:]
    # cond[ii] = 1.
    ii = np.where(np.logical_and(sb > 19.61, sb <= 20.78))[0][:]
    cond[ii] = 0.8
    ii = np.where(np.logical_and(sb > 20.78, sb <= 21.37))[0][:]
    cond[ii] = 0.5
    ii = np.where(sb > 21.37)[0][:]
    cond[ii] = 0.2
    return cond


def sb(mpa, mdist, mZD, ZD, sZD, cc=0.0, verbose = False):
    """
    Calculate sky brightness based on formulas from Krisciunas & Schaefer 1991
    Bryan Miller
    November 5, 2004
    June 1, 2015 - added cc parameter while testing cloud scattering corrections

    Matt Bonnyman
    converted from IDL to Python May 23, 2018

    Parameters
    ----------
    mpa : '~astropy.units.Quantity'
        Moon phase angle at solar midnight in degrees

    mdist : array of '~astropy.units.Quantity'
        Numpy array of angular distances between target and moon

    mZD : array of '~astropy.units.Quantity'
        Numpy array of Moon zenith distance angles

    ZD : array of '~astropy.units.Quantity'
        Numpy array of target zenith distance angles

    sZD : array of '~astropy.units.Quantity'
        Numpy array of Sun zenith distance angles

    cc : array of floats
        Current cloud condition.

    Returns
    ---------
    skybright : float
        Numpy array of sky background magnitudes at target location
    """

    k = 0.172  # mag/airmass relation for Hale Pohaku
    a = 2.51189
    Q = 27.78151

    mpaa = np.asarray(mpa.value)
    if mpaa.ndim == 0:
        mpaa = np.ones(len(mZD))*mpa.value

    sun_alt = 90.0 * u.deg - sZD  # sun altitude
    if verbose:
        print('sun_alt', sun_alt)

    # Dark sky zenith V surface brightness
    Vzen = np.ones(len(ZD)) * 21.587
    # correction for twilight
    ii = np.where(sun_alt > -18.5 * u.deg)[0][:]
    Vzen[ii] = Vzen[ii] - ztwilight(sun_alt[ii])

    # zenith sky brightness
    Bzen = 0.263 * a**(Q - Vzen)
    # Sky brightness with no moon at target, scattering due to airmass
    Bsky = Bzen * xair(ZD) * 10.0**(-0.4 * k * (xair(ZD) - 1.0))

    # Lunar sky brightness
    n = len(Bsky)
    Bmoon = np.zeros(n)

    istar = 10.**(-0.4 * (3.84 + 0.026 * abs(mpaa) + 4.e-9 * mpaa**4.))

    ii = np.where(mZD < 90.8 * u.deg)[0][:]

    jj = ii[np.where(mdist[ii] >= 10. * u.deg)[0][:]]
    if len(jj) != 0:
        fpjj = (1.06 + np.cos(mdist[jj])**2) * 10.0**5.36 + 10.0**(6.15 - mdist[jj].value / 40.0)
        Bmoon[jj] = fpjj * istar[jj] * 10**(-0.4 * k * xair(mZD[jj])) * (1.0 - 10**(-0.4 * k * xair(ZD[jj])))

    kk = np.where(ii != jj)[0][:]
    if len(kk) != 0:
        # There is a bug in the following line from the original code, used by QPT
        fpkk = 6.2e7 / (mdist[kk].value**2)
        # fpkk = (1.06 + np.cos(mdist[kk])**2) * 10.0**5.36 + 6.2e7 / (mdist[kk].value**2)
        Bmoon[kk] = fpkk * istar[kk] * 10**(-0.4 * k * xair(mZD[kk])) * (1.0 - 10**(-0.4 * k * xair(ZD[kk])))

    # hh = np.where(np.logical_and(cc > 0.5, cc < 0.8))[0][:]
    # if len(hh) != 0:  # very simple increase in SB if there are thin clouds
    #     Bmoon[hh] = 2.0 * Bmoon[hh]

    skybright = Q - np.log10((Bmoon + Bsky)/0.263) / np.log10(a)  # sky brightness in Vmag/arcsec^2

    if verbose:
        print('Vzen', Vzen)
        print('Bzen', Bzen)
        print('istar', istar)
        print('Bmoon', Bmoon)
        print('Bsky', Bsky)
        print('skybright', skybright)

    return skybright


def sb2(mpa, mAng, mDist, mZD, ZD, sZD, cc=0.0, verbose=False):
    """
    Calculate sky brightness based on formulas from Krisciunas & Schaefer 1991
    Uses array processing
    Bryan Miller
    November 5, 2004
    June 1, 2015 - added cc parameter while testing cloud scattering corrections
    July 3, 2020 - add terms for lunar distance and close to/on lunar disk,
                   using skycalc approach

    Matt Bonnyman
    converted from IDL to Python May 23, 2018

    Parameters
    ----------
    mpa : '~astropy.units.Quantity'
        Numpy array of Moon phase angle in degrees

    mAng : array of '~astropy.units.Quantity'
        Numpy array of angular distances between target and moon

    mZD : array of '~astropy.units.Quantity'
        Numpy array of Moon zenith distance angles

    mDist : array of '~astropy.units.Quantity'
        Numpy array of distance from the Earth to the Moon

    ZD : array of '~astropy.units.Quantity'
        Numpy array of target zenith distance angles

    sZD : array of '~astropy.units.Quantity'
        Numpy array of Sun zenith distance angles

    cc : array of floats
        Current cloud condition.

    Returns
    ---------
    skybright : float
        Numpy array of sky background magnitudes at target location
    """

    # Constants
    k = 0.172  # mag/airmass relation for Hale Pohaku
    a = 2.51189
    Q = 27.78151
    EQUAT_RAD = 6378137. * u.m  # /* equatorial radius of earth, meters */

    sun_alt = 90.0 * u.deg - sZD  # sun altitude
    if verbose:
        print('sun_alt', sun_alt)

    # Relative distance between the Earth and the Moon
    norm_moondist = mDist / (60.27 * EQUAT_RAD)

    # Number of positions
    n = len(ZD)

    # Dark sky zenith V surface brightness
    Vzen = np.ones(n) * 21.587
    # correction for twilight
    i = np.where(sun_alt > -18.5 * u.deg)[0][:]
    if len(i) != 0:
        Vzen[i] = Vzen[i] - ztwilight(sun_alt[i])

    # zenith sky brightness
    Bzen = 0.263 * a ** (Q - Vzen)
    # Sky brightness with no moon at target, scattering due to airmass
    Bsky = Bzen * xair(ZD) * 10.0 ** (-0.4 * k * (xair(ZD) - 1.0))

    # Lunar sky brightness
    Bmoon = np.zeros(n)
    istar = np.zeros(n)
    frho = np.zeros(n)

    # Do calculations when the Moon is above or near the horizon
    im = np.where(mZD < 90.8 * u.deg)[0][:]

    # Illuminance with phase angle, eq. 8
    istar[im] = 10. ** (-0.4 * (3.84 + 0.026 * abs(mpa[im].value) + 4.e-9 * mpa[im].value ** 4.))

    # Correction for lunar distance
    istar[im] = istar[im] / norm_moondist[im] ** 2

    # Rough correction for opposition effect
    # 35 per cent brighter at full, effect tapering linearly to
    #   zero at 7 degrees away from full. mentioned peripherally in
    #   Krisciunas and Scheafer, p. 1035. */
    hh = im[np.where(abs(mpa[im]) < 7. * u.deg)[0][:]]
    if len(hh) != 0.0:
        istar[hh] *= (1.35 - 0.05 * abs(mpa[hh].value))
    #     if abs(mpa) < 7.*u.deg:
    #         istar[im] *= (1.35 - 0.05 * abs(mpa.value))

    # Build eq. 21 piecewise, mAng is rho in skycalc and the paper
    frho[im] = 229087. * (1.06 + np.cos(mAng[im]) ** 2.)

    # Separation > 10 deg
    ii = im[np.where(abs(mAng[im]) > 10. * u.deg)[0][:]]
    if len(ii) != 0:
        frho[ii] += 10.0 ** (6.15 - mAng[ii].value / 40.0)

    # Separation between 0.25 and 10 deg
    jj = im[np.where(np.logical_and(abs(mAng[im]) > 0.25 * u.deg,
                                    abs(mAng[im]) <= 10. * u.deg))[0][:]]
    if len(jj) != 0:
        frho[jj] += 6.2e7 / (mAng[jj].value ** 2)  # eq 19

    # If on lunar disk
    kk = im[np.where(abs(mAng[im]) <= 0.25 * u.deg)[0][:]]
    if len(kk) != 0:
        frho[kk] += 9.9e8

    # Sky brightnes from the Moon in nanoLamberts (B)
    Bmoon[im] = frho[im] * istar[im] * 10 ** (-0.4 * k * xair(mZD[im])) * \
                (1.0 - 10 ** (-0.4 * k * xair(ZD[im])))

    # hh = np.where(np.logical_and(cc > 0.5, cc < 0.8))[0][:]
    # if len(hh) != 0:  # very simple increase in SB if there are thin clouds
    #     Bmoon[hh] = 2.0 * Bmoon[hh]

    # sky brightness in Vmag/arcsec^2
    skybright = Q - np.log10((Bmoon + Bsky) / 0.263) / np.log10(a)

    if verbose:
        print('Vzen', Vzen)
        print('Bzen', Bzen)
        print('istar', istar)
        print('Bmoon', Bmoon)
        print('Bsky', Bsky)
        print('skybright', skybright)

    return skybright
