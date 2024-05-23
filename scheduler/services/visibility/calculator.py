import json
import time
from dataclasses import dataclass
from typing import final, Dict, List, Any

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta, Time
from lucupy import sky
from lucupy.decorators import immutable
from lucupy.timeutils import time2slots
from lucupy.minimodel import SiderealTarget, NonsiderealTarget, SkyBackground, ElevationType, Constraints, NightIndex, \
    Observation, Target, Program
from numpy import dtype, ndarray

from scheduler.services.proper_motion import ProperMotionCalculator
from scheduler.services.ephemeris import EphemerisCalculator
from scheduler.services.redis import redis_client

from .snapshot import VisibilitySnapshot, TargetSnapshot
from ..resource import NightConfiguration
from ...core.calculations import NightEvents


@final
@immutable
@dataclass(frozen=True)
class TargetVisibility:
    visibility_slot_idx: npt.NDArray[int]
    visibility_time: TimeDelta
    rem_visibility_time: TimeDelta
    rem_visibility_frac: float


def calculate_target_snapshot(night_idx: NightIndex,
                              obs: Observation,
                              target: Target,
                              night_events: NightEvents,
                              time_grid_night: Time,
                              time_slot_length: TimeDelta):
    """
    Calculate the target information for a period of time.
    """

    # Calculate the ra and dec for each target.
    # In case we decide to go with numpy arrays instead of SkyCoord,
    # this information is already stored in decimal degrees at this point.
    num_time_slots = night_events.num_timeslots_per_night[night_idx]
    match target:
        case SiderealTarget() as sidereal_target:
            coord = ProperMotionCalculator().calculate_coordinates(sidereal_target,
                                                                   time_grid_night,
                                                                   num_time_slots,
                                                                   time_slot_length)
        case NonsiderealTarget() as nonsidereal_target:

            sunset = night_events.sunset[night_idx]
            sunrise = night_events.sunrise[night_idx]
            eph_coord = EphemerisCalculator().calculate_coordinates(obs.site,
                                                                    nonsidereal_target,
                                                                    sunset,
                                                                    sunrise)

            # Now trim the coords to the desired subset.
            int_time_slot_length = int(time_slot_length.to_datetime().total_seconds() / 60)
            sunset_to_twi = night_events.twilight_evening_12[night_idx] - sunset
            start_time_slot = time2slots(time_slot_length.to_datetime(), sunset_to_twi.to_datetime())
            end_time_slot = start_time_slot + num_time_slots

            # We must take every x minutes where x is the time slot length in minutes.
            coord = eph_coord[start_time_slot:end_time_slot:int_time_slot_length]
        case _:
            msg = f'Invalid target: {target}'
            raise ValueError(msg)

    # Calculate the hour angle, altitude, azimuth, parallactic angle, and airmass.
    lst = night_events.local_sidereal_times[night_idx]
    # TODO: Remove debugging
    # print(f'Night idx: {night_idx}, num time slots: {lst.size}')

    hourangle = lst - coord.ra
    hourangle.wrap_at(12.0 * u.hour, inplace=True)
    alt, az, par_ang = sky.Altitude.above(coord.dec, hourangle, obs.site.location.lat)
    airmass = sky.true_airmass(alt)

    # Determine time slot indices where the sky brightness and elevation constraints are met.
    # By default, in the case where an observation has no constraints, we use SB ANY.
    # TODO: moon_dist here is a List[float], when calculate_sky_brightness expects a Distance.
    # TODO: code still works, bt we should be very careful here.
    if obs.constraints and obs.constraints.conditions.sb < SkyBackground.SBANY:
        targ_sb = obs.constraints.conditions.sb
        targ_moon_ang = coord.separation(night_events.moon_pos[night_idx])
        brightness = sky.brightness.calculate_sky_brightness(
            180.0 * u.deg - night_events.sun_moon_ang[night_idx],
            targ_moon_ang,
            night_events.moon_dist[night_idx],
            90.0 * u.deg - night_events.moon_alt[night_idx],
            90.0 * u.deg - alt,
            90.0 * u.deg - night_events.sun_alt[night_idx]
        )
        sb = sky.brightness.convert_to_sky_background(brightness)
    else:
        targ_sb = SkyBackground.SBANY
        sb = np.full([len(night_events.times[night_idx])], SkyBackground.SBANY)

    return TargetSnapshot(coord=coord,
                          alt=alt,
                          az=az,
                          par_ang=par_ang,
                          hourangle=hourangle,
                          airmass=airmass,
                          target_sb=targ_sb,
                          sky_brightness=sb)


def calculate_target_visibility(obs: Observation,
                                target: Target,
                                prog: Program,
                                night_events: NightEvents,
                                nc: dict[ndarray[Any, dtype[NightIndex]], NightConfiguration],
                                time_grid: Time,
                                timing_windows: List[Time],
                                time_slot_length: TimeDelta) -> Dict[NightIndex,TargetVisibility]:
    """
    Iterate over the time grid, checking to see if there is already a TargetInfo
    for the target for the given day at the given site.
    If so, we skip.
    If not, we execute the calculations and store.
    In order to properly calculate the:
    * rem_visibility_time: total time a target is visible from the current night to the end of the period
    * rem_visibility_frac: fraction of remaining observation length to rem_visibility_time
    we want to process the nights BACKWARDS so that we can sum up the visibility time.
    """

    rem_visibility_time = 0.0 * u.h
    rem_visibility_frac_numerator = obs.exec_time() - obs.total_used()

    target_visibilities: Dict[NightIndex, TargetVisibility] = {}

    visibility_snapshots: Dict[str, Dict] = {}

    key = f'{obs.id.id}{time_slot_length}'
    exists = redis_client.exists(key)
    if exists:
        visibility_snapshots = json.loads(redis_client.get(key))
    else:
        for ridx, jday in enumerate(reversed(time_grid)):
            # Convert to the actual time grid index.
            night_idx = NightIndex(len(time_grid) - ridx - 1)
            # Calculate the time slots for the night in which there is visibility.
            visibility_slot_idx = np.array([], dtype=int)

            # Calculate target snapshot
            target_snapshot = calculate_target_snapshot(night_idx,
                                                        obs,
                                                        target,
                                                        night_events,
                                                        time_grid[night_idx],
                                                        time_slot_length)
            # In the case where an observation has no constraint information or an elevation constraint
            # type of None, we use airmass default values.
            if obs.constraints and obs.constraints.elevation_type != ElevationType.NONE:
                targ_prop = target_snapshot.hourangle if obs.constraints.elevation_type is ElevationType.HOUR_ANGLE else target_snapshot.airmass
                elev_min = obs.constraints.elevation_min
                elev_max = obs.constraints.elevation_max
            else:
                targ_prop = target_snapshot.airmass
                elev_min = Constraints.DEFAULT_AIRMASS_ELEVATION_MIN
                elev_max = Constraints.DEFAULT_AIRMASS_ELEVATION_MAX

            # Are all the required resources available?
            # This works for validation mode. In RT mode, this may need to be statistical if resources are not known
            # and they could change with time, so the visfrac calc may need to be extracted from this method
            has_resources = all([resource in nc[night_idx].resources for resource in obs.required_resources()])
            avail_resources = np.full([len(night_events.times[night_idx])], int(has_resources), dtype=int)

            # Is the program excluded on a given night due to block scheduling
            can_schedule = nc[night_idx].filter.program_filter(prog)
            is_schedulable = np.full([len(night_events.times[night_idx])], int(can_schedule), dtype=int)
            # print(f"{obs.unique_id} {has_resources} {can_schedule}")

            # Calculate the time slot indices for the night where:
            # 1. The sun altitude requirement is met (precalculated in night_events)
            # 2. The sky background constraint is met
            # 3. The elevation constraints are met
            sa_idx = night_events.sun_alt_indices[night_idx]

            c_idx = np.where(
                np.logical_and(target_snapshot.sky_brightness[sa_idx] <= target_snapshot.target_sb,
                               np.logical_and(avail_resources[sa_idx] == 1,
                                              np.logical_and(is_schedulable[sa_idx] == 1,
                                                             np.logical_and(targ_prop[sa_idx] >= elev_min,
                                                                            targ_prop[sa_idx] <= elev_max))))
            )[0]

            # Apply timing window constraints.
            # We always have at least one timing window. If one was not given, the program length will be used.
            for tw in timing_windows:
                tw_idx = np.where(
                    np.logical_and(night_events.times[night_idx][sa_idx[c_idx]] >= tw[0],
                                   night_events.times[night_idx][sa_idx[c_idx]] <= tw[1])
                )[0]
                visibility_slot_idx = np.append(visibility_slot_idx, sa_idx[c_idx[tw_idx]])

            # Create a visibility filter that has an entry for every time slot over the night,
            # with 0 if the target is not visible and 1 if it is visible.
            visibility_slot_filter = np.zeros(len(night_events.times[night_idx]))
            visibility_slot_filter.put(visibility_slot_idx, 1.0)

            # TODO: Guide star availability for moving targets and parallactic angle modes.

            # Calculate the visibility time, the ongoing summed remaining visibility time, and
            # the remaining visibility fraction.
            # If the denominator for the visibility fraction is 0, use a value of 0.
            visibility_time = len(visibility_slot_idx) * time_slot_length

            visibility_snapshot = VisibilitySnapshot(visibility_slot_idx=visibility_slot_idx,
                                                     visibility_time=visibility_time)
            # Pass to int to eliminate decimals and to string to keep the keys after deserialization.
            visibility_snapshots[str(int(jday.jd))] = visibility_snapshot.to_dict()
        redis_client.set(key, json.dumps(visibility_snapshots))

    for ridx, jday in enumerate(reversed(time_grid)):
        # Convert to the actual time grid index.
        night_idx = NightIndex(len(time_grid) - ridx - 1)
        visibility_snapshot = VisibilitySnapshot.from_dict(visibility_snapshots[str(int(jday.jd))])

        rem_visibility_time += visibility_snapshot.visibility_time
        if rem_visibility_time.value:
            # This is a fraction, so convert to seconds to cancel the units out.
            rem_visibility_frac = (rem_visibility_frac_numerator.total_seconds() /
                                   rem_visibility_time.to_value(u.s))
        else:
            rem_visibility_frac = 0.0

        target_visibilities[night_idx] = TargetVisibility(visibility_slot_idx=visibility_snapshot.visibility_slot_idx,
                                                          visibility_time=visibility_snapshot.visibility_time,
                                                          rem_visibility_time=rem_visibility_time,
                                                          rem_visibility_frac=rem_visibility_frac)
    return target_visibilities
