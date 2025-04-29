#!/usr/bin/env python

# Example for how to compute the hours that a target is visible on a night from Collector information
# Bryan Miller
# 2025apr09

from pathlib import Path
from typing import Optional

from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle
import astropy.units as u

from lucupy.minimodel.site import ALL_SITES
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.sky.utils import true_airmass, min_max_alt, local_sidereal_time, alt_to_hour_angle
from lucupy.minimodel import Group, ObservationID, GroupID, UniqueGroupID, ProgramID, QAState, ObservationClass, ObservationStatus, Band
# from lucupy.minimodel import Site, NightIndex, VariantSnapshot, TimeslotIndex
# from lucupy.timeutils import time2slots

from definitions import ROOT_DIR
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters
# from scheduler.engine import SchedulerParameters, Engine
from scheduler.services import logger_factory
# from scheduler.services.visibility import visibility_calculator

# from scheduler.core.output import print_collector_info, print_plans
from scheduler.engine.params import SchedulerParameters
# from scheduler.engine.scp import SCP

from scheduler.core.builder import Blueprints
from scheduler.core.builder.modes import dispatch_with
# from scheduler.core.builder.blueprint import OptimizerBlueprint
# from scheduler.core.components.changemonitor import ChangeMonitor, TimeCoordinateRecord
# from scheduler.core.components.ranker import DefaultRanker
from scheduler.core.eventsqueue import EventQueue, EveningTwilightEvent, WeatherChangeEvent, MorningTwilightEvent, Event
# from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
# from scheduler.core.plans import Plans
from scheduler.core.sources import Sources
# from scheduler.core.sources.origins import Origins
# from scheduler.core.statscalculator import StatCalculator

import numpy as np
from matplotlib import pyplot as plt

_logger = logger_factory.create_logger(__name__)


def alt_from_airmass(airmass: float, plot=False) -> Angle:
    """Return the altitude(s) corresponding to a true airmass

    Parameters
    ----------
    airmass : float, can be array
        True airmasses for which altitudes are needed.
    """
    airmass = np.asarray(airmass)
    scalar_input = False
    if airmass.ndim == 0:
        airmass = airmass[None]
        scalar_input = True

    # Calculate true airmasses for different altitudes
    a = Angle(np.arange(90, 0, -0.1), unit=u.deg)
    x_true = true_airmass(a)

    # Interpolate the altitudes for the given airmasses
    alt = np.interp(airmass, x_true, a)

    if plot:
        plt.plot(x_true, a)
        for ii in range(len(airmass)):
            plt.axvline(airmass[ii])
            plt.axhline(alt[ii].value)
        plt.ylabel('Altitude')
        plt.xlabel('Airmass')
        plt.xlim(0.95, 3)
        plt.show()

    if scalar_input:
        return np.squeeze(alt)
    return alt


def hrs_up(timeup, timedown, evening, morning):
    """returns a TimeDelta giving how long an object is up during the
    night, basically the intersection between the interval is's 'up' above
    a given altitude, and the interval when it's nighttime.

    Checks are implemented for circumpolar objects that can set and come back up.

    Parameters
    ----------
    timeup : astropy Time, can be array
        Time at which the object rises above an altitude
    timedown : astropy Time, can be array of the same size as timeup
        Time at which an object sets below a given altitude
    evening : astropy Time, can be array of the same size as timeup
        Time of the beginning of the night
    morning : astropy Time, can be array of the same size as timeup
        Time of the ending of the night
    """

    # all are Times.
    #  timeup - when the object rises past a given altitle
    #  timedown - when the object sets past a given altitle
    #  evening - time of evening twiligth (however defined)
    #  morning - time of morning twilight
    # return value will be a TimeDelta

    timeup = Time(np.asarray(timeup.jd), format='jd')
    scalar_input = False
    if timeup.ndim == 0:
        timeup = timeup[None]
        scalar_input = True
    timedown = Time(np.asarray(timedown.jd), format='jd')
    if timedown.ndim == 0:
        timedown = timedown[None]
    evening = Time(np.asarray(evening.jd), format='jd')
    if evening.ndim == 0:
        evening = evening[None]
    morning = Time(np.asarray(morning.jd), format='jd')
    if morning.ndim == 0:
        morning = morning[None]

    if len(timeup) != len(timedown):
        print('Error: timeup and timedown must have the same dimensions')
        return None

    hrsup = TimeDelta(0.0 * np.zeros(len(timeup)), format='jd')

    # up all night
    ii = np.where(np.logical_and(timeup < evening, timedown > morning))[0][:]
    if len(ii) != 0:
        hrsup[ii] = morning[ii] - evening[ii]

    # up and down the same night
    ii = np.where(np.logical_and(timeup >= evening, timedown <= morning))[0][:]
    if len(ii) != 0:
        # rise then set
        jj = ii[np.where(timedown[ii] >= timeup[ii])[0][:]]
        if len(jj) != 0:
            hrsup[jj] = timedown[jj] - timeup[jj]
        # set then rise the same night
        kk = ii[np.where(timeup[ii] > timedown[ii])[0][:]]
        if len(kk) != 0:
            hrsup[kk] = (timedown[kk] - evening[kk]) + (morning[kk] - timeup[kk])

    # Rise before evening
    ii = np.where(np.logical_and(timeup < evening, timedown <= morning))[0][:]
    if len(ii) != 0:
        hrsup[ii] = timedown[ii] - evening[ii]

    # Set after morning
    ii = np.where(np.logical_and(timeup >= evening, timedown > morning))[0][:]
    if len(ii) != 0:
        hrsup[ii] = morning[ii] - timeup[ii]

    if scalar_input:
        hrsup = np.squeeze(hrsup)
    return hrsup


def hours_up(obsid: ObservationID, night_idx: int, airmass_max=2.05):
    """Calculate the hours that a target is visible above a given altitude on a given night (night_idx)"""

    # Get target info from observation
    obs = collector.get_observation(obsid)
    ti = collector.get_target_info(obs.id)
    # ra = ti[0].coord.ra[0]
    # dec = ti[0].coord.dec[0]
    ra = ti[0].coord.ra
    dec = ti[0].coord.dec
    # print(ra, dec)

    # Get the site info
    site = obs.site
    # print(site)
    for s in collector.sites:
        if s == obs.site:
            site = s
    minalt, maxalt = min_max_alt(site.location.lat, dec)
    # print(minalt, maxalt)

    # Night events (twilights and midnight)
    night_events = collector.get_night_events(site)
    teventwi12 = night_events.twilight_evening_12[night_idx]
    tmorntwi12 = night_events.twilight_morning_12[night_idx]
    # print(teventwi12, tmorntwi12)

    midnight = night_events.midnight[night_idx]
    # print(f'Midnight UT: {midnight}')
    # LST of midnight
    lstmid = local_sidereal_time(midnight, site.location)
    # print(f'Midnight LST: {lstmid}')

    # Hour angle of target at midnight
    ha_mid = lstmid - ra.wrap_at(12. * u.hour)
    # print(ha_mid)

    # Time of transit (crossing the meridian)
    deltattran = TimeDelta(ha_mid.hourangle / 24., format='jd') / 1.0027379093
    ttransit = midnight - deltattran

    # Altitude corresponding to the airmass limit desired
    # ALT_LOW           = 29.8796 * u.deg # altitude at which true airm = 2.0
    ALT_LOW = alt_from_airmass(airmass_max)

    if minalt < ALT_LOW and maxalt > ALT_LOW:
        ha = alt_to_hour_angle(dec, site.location.lat, ALT_LOW)
        dt = TimeDelta(ha.hourangle / 24., format='jd') / 1.0027379093
        jd_1 = ttransit - dt
        jd_2 = ttransit + dt
        uptime = hrs_up(jd_1, jd_2, teventwi12, tmorntwi12)
    elif minalt > ALT_LOW:
        uptime = (tmorntwi12 - teventwi12)
    elif maxalt < ALT_LOW:
        uptime = TimeDelta(0., format='jd')

    return uptime.sec / 3600.


# Set lucupy to Gemini
ObservatoryProperties.set_properties(GeminiProperties)

programs_ids = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.txt'
# programs_ids = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.redis.txt'

# Parsed program file (this replaces the program picker from Schedule)
with open(programs_ids, 'r') as file:
    programs_list = [line.strip() for line in file if line.strip()[0] != '#']

## Collector
# Initial variables
verbose = False
test_events = True

# Create Parameters
params = SchedulerParameters(start=Time("2018-10-01 08:00:00", format='iso', scale='utc'),
                                 end=Time("2018-10-03 08:00:00", format='iso', scale='utc'),
                                 sites=ALL_SITES,
                                 mode=SchedulerModes.VALIDATION,   #Options: VALITADION, SIMULATION, OPERATION
                                 ranker_parameters=RankerParameters(),
                                 semester_visibility=False,
                                 num_nights_to_schedule=1,
                                 programs_list=programs_list)

# Engine.build
sources = Sources()

# Create event queue to handle incoming events.
queue = EventQueue(params.night_indices, params.sites)
# Create builder based in the mode to create SCP
builder = dispatch_with(params.mode, sources, queue)

# Create Collector
collector = builder.build_collector(start=params.start,
                                            end=params.end_vis,
                                            num_of_nights=params.num_nights_to_schedule,
                                            sites=params.sites,
                                            semesters=params.semesters,
                                            blueprint=Blueprints.collector,
                                            program_list=params.programs_list)

# Example observation with visibility
obs_vis = collector.get_observation(ObservationID('GN-2018B-Q-103-16'))
ti_vis = collector.get_target_info(obs_vis.id)
# print(ti_vis[1].visibility_slot_idx)

# Timeslot length in minutes
timeslot_min = collector.time_slot_length * 24 * 60

# Visibility from visibility_slot_idx, includes SB constraints, timing windows, resources, airmass constraints, etc
print(f'Visibility from slots_idx: {len(ti_vis[1].visibility_slot_idx) * timeslot_min} min')

# Hours below airmass 2.05 between nautical twilights based on hour angle and time
up = hours_up(ObservationID('GN-2018B-Q-103-16'), 1)
print(f'Uptime < airmass 2.05: {up * 60} min')

# Check with target information airmass array, this should match hours_up to within one time slot
ia = np.where(ti_vis[1].airmass <= 2.05)[0]
print(f'Airmass < 2.05: {len(ia) * timeslot_min} min')

# Conclusion, the easiest and most accurate way to determine if an observation is scheduleable on a given night
# is to see if the visibility_slot_idx array has non-zero length.

plt.plot(ti_vis[1].airmass)
plt.plot(ti_vis[1].visibility_slot_idx, ti_vis[1].airmass[ti_vis[1].visibility_slot_idx], linewidth=4)
plt.ylim(3.0, 0.95)
plt.show()










