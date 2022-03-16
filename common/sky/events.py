import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time, TimeDelta
from sky.utils import local_sidereal_time, hour_angle_to_angle, local_midnight_time
from sky.sun import Sun
from sky.constants import EQUAT_RAD
from sky.moon import Moon



def night_events(aTime: Time, location: EarthLocation, localtzone, verbose=True):
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
    sunmid = Sun.location(midnight)
    sunpos = Sun.location(aTime)

    # allow separate rise and set altitudes for horizon effects
    horiz = (-0.883 - np.sqrt(2. * location.height / EQUAT_RAD) * (180. / np.pi)) * u.deg
    # print(horiz)
    setalt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    risealt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    twialt12 = Angle(-12. * np.ones(nt), unit=u.deg)  # 12 degree nautical twilight

    # print(sunmid.dec, setalt)
    sunsetha = hour_angle_to_angle(sunmid.dec, location.lat, setalt)  # corresponding hr angles
    sunriseha = Angle(2. * np.pi, unit=u.rad) - hour_angle_to_angle(sunmid.dec, location.lat, risealt)  # corresponding hr angles
    twilightha12 = hour_angle_to_angle(sunmid.dec, location.lat, twialt12)

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
    tsunset = Sun.time_by_altitude(setalt, tsunset, location)

    tsunrise = midnight + TDsunrise  # first approx
    tsunrise = Sun.time_by_altitude(risealt, tsunrise, location)

    tevetwi12 = midnight - TDevetwi12
    tevetwi12 = Sun.time_by_altitude(twialt12, tevetwi12, location)

    tmorntwi12 = midnight + TDmorntwi12
    tmorntwi12 = Sun.time_by_altitude(twialt12, tmorntwi12, location)

    if verbose:
        print("sunset: ", tsunset)
        print("sunrise: ", tsunrise)
        print("eve twi12: ", tevetwi12)
        print("morn twi12:", tmorntwi12)
    # Moon
    moonmid = Moon.low_precision_location(midnight, location)
    hamoonmid = lstmid - moonmid.ra
    hamoonmid.wrap_at(12. * u.hour, inplace=True)
    moonpos, topodist = Moon.time_by_altitude(aTime, location)

    if verbose:
        print("moon at midnight: ", moonmid.to_string('hmsdms'))
        print("hamoonmid: ", hamoonmid.hour, 'hr')

    # roughlunarday = TimeDelta(1.0366, format='jd')

    moonsetha = hour_angle_to_angle(moonmid.dec, location.lat, setalt)  # corresponding hr angles
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
    tmoonset = (setalt, tmoonset, location)
    # if verbose: print("moonset: ", tmoonset)
    if verbose: print("moonset: ", tmoonset)

    moonriseha = -1. * hour_angle_to_angle(moonmid.dec, location.lat, risealt)  # signed
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
    tmoonrise = Moon.time_by_altitude(risealt, tmoonrise, location)
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
