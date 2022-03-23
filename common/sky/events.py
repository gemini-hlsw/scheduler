import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time, TimeDelta, TimezoneInfo
from common.sky.utils import local_sidereal_time, hour_angle_to_angle, local_midnight_time
from common.sky.sun import Sun
from common.sky.constants import EQUAT_RAD
from common.sky.moon import Moon



def night_events(time: Time, location: EarthLocation, localtzone: TimezoneInfo, verbose=True):
    """
    Compute phenomena for a given night.

    This is mostly a testbed that prints results directly.

    Parameters
    ----------
    time : astropy Time, single or array
        input time; if before noon, events of previous night are computed.
    location : EarthLocation
    localtzone : timezone object.
    verbose: verbose output
    """
    # prototype for the events of a single night -- sunset and rise,
    # twilights, and moonrise and set.

    time = Time(np.asarray(time.iso), format='iso')
    scalar_input = False
    if time.ndim == 0:
        time = time[None]  # Makes 1D
        scalar_input = True

    nt = len(time)

    midnight = local_midnight_time(time, localtzone)  # nearest clock-time midnight (UT)
    lst_midnight = local_sidereal_time(midnight, location)

    # Sun
    sun_at_midnight = Sun.at(midnight)

    # allow separate rise and set altitudes for horizon effects
    horiz = (-0.883 - np.sqrt(2. * location.height / EQUAT_RAD) * (180. / np.pi)) * u.deg

    set_alt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    rise_alt = Angle(horiz * np.ones(nt), unit=u.deg)  # zd = 90 deg 50 arcmin
    twelve_twilight_alt = Angle(-12. * np.ones(nt), unit=u.deg)  # 12 degree nautical twilight

    sunset_ha = hour_angle_to_angle(sun_at_midnight.dec, location.lat, set_alt)  # corresponding hr angles
    sunrise_ha = Angle(2. * np.pi, unit=u.rad) - hour_angle_to_angle(sun_at_midnight.dec, location.lat, rise_alt)  # corresponding hr angles
    twelve_twilight_ha = hour_angle_to_angle(sun_at_midnight.dec, location.lat, twelve_twilight_alt)
    sun_at_midnight_ha = (lst_midnight - sun_at_midnight.ra).wrap_at(24. * u.hour)

    sunset_guess = sun_at_midnight_ha - sunset_ha  # angles away from midnight
    sunrise_guess = sunrise_ha - sun_at_midnight_ha
    even_12twi_guess = sun_at_midnight_ha - twelve_twilight_ha
    morn_12twi_guess = Angle(2. * np.pi, unit=u.rad) - twelve_twilight_ha - sun_at_midnight_ha

    # convert to time deltas
    timedelta_sunset = TimeDelta(sunset_guess.hour / 24., format='jd')
    timedelta_sunrise = TimeDelta(sunrise_guess.hour / 24., format='jd')
    timedelta_even_12twi = TimeDelta(even_12twi_guess.hour / 24., format='jd')
    timedelta_morn_12twi = TimeDelta(morn_12twi_guess.hour / 24., format='jd')
    
    # form into times and iterate to accurate answer.
    times_sunset = midnight - timedelta_sunset  # first approx
    times_sunset = Sun.time_by_altitude(set_alt, times_sunset, location)

    times_sunrise = midnight + timedelta_sunrise  # first approx
    times_sunrise = Sun.time_by_altitude(rise_alt, times_sunrise, location)

    times_even_12twi = midnight - timedelta_even_12twi
    times_even_12twi = Sun.time_by_altitude(twelve_twilight_alt, times_even_12twi, location)

    times_morn_12twi = midnight + timedelta_morn_12twi
    times_morn_12twi = Sun.time_by_altitude(twelve_twilight_alt, times_morn_12twi, location)

    if verbose:
        print("sunset: ", times_sunset)
        print("sunrise: ", times_sunrise)
        print("eve twi12: ", times_even_12twi)
        print("morn twi12:", times_morn_12twi)

    # Moon
    moon_at_midnight, _ = Moon().at(midnight).low_precision_location(location)
    ha_moon_at_midnight = lst_midnight - moon_at_midnight.ra
    ha_moon_at_midnight.wrap_at(12. * u.hour, inplace=True)

    if verbose:
        print("moon at midnight: ", moon_at_midnight.to_string('hmsdms'))
        print("ha_moon_at_midnight: ", ha_moon_at_midnight.hour, 'hr')

    ha_moon_set = hour_angle_to_angle(moon_at_midnight.dec, location.lat, set_alt)  # corresponding hr angles
    diff_moon_set = ha_moon_set - ha_moon_at_midnight  # how far from setting point at midn.
    # find nearest setting point
    # if diff_moon_set.hour >= 12. : diff_moon_set = diff_moon_set - Angle(24. * u.hour)
    ii = np.where(diff_moon_set.hour >= 12.)[0][:]
    if len(ii) != 0:
        diff_moon_set[ii] = diff_moon_set[ii] - Angle(24. * u.hour)

    # if diff_moon_set.hour < -12. : diff_moon_set = diff_moon_set + Angle(24. * u.hour)
    jj = np.where(diff_moon_set.hour < -12.)[0][:]
    if len(jj) != 0:
        diff_moon_set[jj] = diff_moon_set[jj] + Angle(24. * u.hour)

    timedelta_moon_set = TimeDelta(diff_moon_set.hour / 24., format='jd')
    times_moon_set = midnight + timedelta_moon_set
    if verbose:
        print("moonset first approx:", times_moon_set)
    times_moon_set = (set_alt, times_moon_set, location)
    # if verbose: print("moonset: ", times_moon_set)
    if verbose:
        print("moonset: ", times_moon_set)

    ha_moonrise = -1. * hour_angle_to_angle(moon_at_midnight.dec, location.lat, rise_alt)  # signed
    diff_moonrise = ha_moonrise - ha_moon_at_midnight  # how far from riseting point at midn.
    # find nearest riseting point
    # if diff_moonrise.hour >= 12.: diff_moonrise = diff_moonrise - Angle(24. * u.hour)
    # if diff_moonrise.hour < -12.: diff_moonrise = diff_moonrise + Angle(24. * u.hour)
    ii = np.where(diff_moonrise.hour >= 12.)[0][:]
    if len(ii) != 0:
        diff_moonrise[ii] = diff_moonrise[ii] - Angle(24. * u.hour)
    jj = np.where(diff_moonrise.hour < -12.)[0][:]
    if len(jj) != 0:
        diff_moonrise[jj] = diff_moonrise[jj] + Angle(24. * u.hour)

    timedelta_moonrise = TimeDelta(diff_moonrise.hour / 24., format='jd')
    times_moonrise = midnight + timedelta_moonrise
    if verbose:
        print("moonrise first approx:", times_moonrise)
    times_moonrise = Moon.time_by_altitude(rise_alt, times_moonrise, location)
    # if verbose: print("moonrise: ", times_moonrise)
    if verbose:
        print("moonrise: ", times_moonrise)

    if scalar_input:
        times_sunset = np.squeeze(times_sunset)
        times_sunrise = np.squeeze(times_sunrise)
        times_even_12twi = np.squeeze(times_even_12twi)
        times_morn_12twi = np.squeeze(times_morn_12twi)
        times_moonrise = np.squeeze(times_moonrise)
        times_moon_set = np.squeeze(times_moon_set)

    return midnight, times_sunset, times_sunrise, times_even_12twi, times_morn_12twi, times_moonrise, times_moon_set
