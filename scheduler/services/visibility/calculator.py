from dataclasses import dataclass
from typing import final, Dict, List, Any, Optional, ClassVar, FrozenSet

import os
import psutil

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.time import TimeDelta, Time
from lucupy import sky
from lucupy.decorators import immutable
from lucupy.timeutils import time2slots
from lucupy.minimodel import SiderealTarget, NonsiderealTarget, SkyBackground, ElevationType, Constraints, NightIndex, \
    Observation, Target, Program, ResourceType, Semester, ObservationID, SemesterHalf
from numpy import dtype, ndarray
from datetime import timedelta

# from scheduler.config import config
from scheduler.core.calculations import NightEvents
from scheduler.core.calculations.targetinfo import TargetInfo, TargetInfoMap, TargetInfoNightIndexMap
from scheduler.services.proper_motion import ProperMotionCalculator
from scheduler.services.ephemeris import EphemerisCalculator
from scheduler.services.redis_client import redis_client
from scheduler.services.logger_factory import create_logger
from scheduler.services.resource import NightConfiguration

from .snapshot import VisibilitySnapshot, TargetSnapshot
from scheduler.core.meta import Singleton

from scheduler.config import config

_logger = create_logger(__name__, with_id=False)

__all__ = [
    'calculate_target_snapshot',
    'program_obs_vis',
    'calculate_target_info',
    'visibility_calculator',
    'TargetVisibility',
    'TargetVisibilityTable',
    'get_cores'
]


@final
# @immutable
@dataclass(frozen=False)
class TargetVisibility:
    visibility_slot_idx: npt.NDArray[int]
    visibility_time: TimeDelta
    rem_visibility_time: TimeDelta
    rem_visibility_frac: float

@final
@dataclass
class TargetVisibilityTable:
    vis_table: Dict[Semester, Dict[str, Dict[str, str]]]

    def get(self, semester: Semester, obs_id: ObservationID, day: str) -> dict:
        """
        Retrieves a VisibilitySnapshot for an observation in a
        given night (in julian date) for the specific semester
        """
        current = self.vis_table
        for key in [semester, obs_id.id, day]:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                if current is None:
                    raise KeyError(f'Missing {key} in Visibility table.')

        return current

    def get_obs(self, semester: Semester, obs_id: ObservationID) -> dict:
        """
        Retrieves a VisibilitySnapshot for an observation in a
        given night (in julian date) for the specific semester
        """
        current = self.vis_table
        for key in [semester, obs_id.id]:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                if current is None:
                    raise KeyError(f'Missing {key} in Visibility table.')

        return current

def execute(command):
    exe = os.popen(command)
    output = exe.readlines()
    exe.close()
    return output


def get_cores():
    """Get CPU and system information"""
    core_info = {'threads': psutil.cpu_count(logical=True), 'cores': psutil.cpu_count(logical=False), 'performance': 0,
                 'efficiency': 0, 'sys': os.uname().sysname, 'arch': execute('uname -m')[0].strip()}

    if 'arm' in core_info['arch'] and 'Darwin' in core_info['sys']:
        # Performance cores
        command = '/usr/sbin/sysctl -n hw.perflevel0.logicalcpu_max'
        core_info['performance'] = int(execute(command)[0].strip())

        # Efficiency cores
        command = '/usr/sbin/sysctl -n hw.perflevel1.logicalcpu_max'
        core_info['efficiency'] = int(execute(command)[0].strip())

    return core_info


def calculate_target_snapshot(night_idx: NightIndex,
                              obs: Observation,
                              target: Target,
                              night_events: NightEvents,
                              time_grid_night: Time,
                              time_slot_length: TimeDelta,
                              for_vis_calc: bool = True):
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
            # Horizons queries are done using a 1min timeslot from sunset to sunrise, then resampled below
            # Get timeslots in the Horizons list between twilights
            horizons_timeslot = timedelta(seconds=60)
            start_time_slot = time2slots(horizons_timeslot, sunset_to_twi.to_datetime())
            end_time_slot = start_time_slot + num_time_slots * int_time_slot_length

            # Choose times between twilights
            coord = eph_coord[start_time_slot:end_time_slot:int_time_slot_length]
        case _:
            msg = f'Invalid target: {target}'
            raise ValueError(msg)

    # Calculate the hour angle, altitude, azimuth, parallactic angle, and airmass.
    lst = night_events.local_sidereal_times[night_idx]

    hourangle = lst - coord.ra
    hourangle.wrap_at(12.0 * u.hour, inplace=True)
    alt, az, par_ang = sky.Altitude.above(coord.dec, hourangle, obs.site.location.lat)
    airmass = sky.true_airmass(alt)


    if for_vis_calc:
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
    else:
        return TargetSnapshot(coord=coord,
                              alt=alt,
                              az=az,
                              hourangle=hourangle,
                              airmass=airmass)

# From calculator.py
def calculate_visibility(obs: Observation,
                         target: Target,
                         prog: Program,
                         night_events: NightEvents,
                         nc: dict[ndarray[Any, dtype[NightIndex]], NightConfiguration],
                         time_grid: Time,
                         timing_windows: List[Time],
                         time_slot_length: TimeDelta,
                         for_vis_calc: bool = True) -> Dict[str, VisibilitySnapshot]:
    """ Iterate over the time grid, checking to see if there is already a TargetInfo
                for the target for the given day at the given site.
                If so, we skip.
                If not, we execute the calculations and store.
                In order to properly calculate the:
                * rem_visibility_time: total time a target is visible from the current night to the end of the period
                * rem_visibility_frac: fraction of remaining observation length to rem_visibility_time
                we want to process the nights BACKWARDS so that we can sum up the visibility time.
    """

    if for_vis_calc:
        visibility_snapshots: Dict[str, VisibilitySnapshot] = {}
    else:
        visibility_snapshots = None
    target_snapshots: Dict[NightIndex, TargetSnapshot] = {}

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
                                                    time_slot_length,
                                                    for_vis_calc=for_vis_calc)
        target_snapshots[night_idx] = target_snapshot

        # In the case where an observation has no constraint information or an elevation constraint
        # type of None, we use airmass default values.
        if for_vis_calc:
            if obs.constraints and obs.constraints.elevation_type != ElevationType.NONE:
                targ_prop = target_snapshot.hourangle.deg if obs.constraints.elevation_type is ElevationType.HOUR_ANGLE else target_snapshot.airmass
                elev_min = obs.constraints.elevation_min
                elev_max = obs.constraints.elevation_max
            else:
                targ_prop = target_snapshot.airmass
                elev_min = Constraints.DEFAULT_AIRMASS_ELEVATION_MIN
                elev_max = Constraints.DEFAULT_AIRMASS_ELEVATION_MAX

            # Are all the required resources available?
            # This works for validation mode. In RT mode, this may need to be statistical if resources are not known
            # and they could change with time, so the visfrac calc may need to be extracted from this method
            if "GMOS" in obs.instrument().id:
                has_resources = all([resource in nc[night_idx].resources for resource in obs.required_resources()])
            else:
                has_resources = all([resource in nc[night_idx].resources for resource in obs.required_resources() if resource.type != ResourceType.FILTER and resource.type != ResourceType.DISPERSER and resource.type != ResourceType.FPU])

            if not has_resources:
                visibility_snapshots[str(int(jday.jd))] = VisibilitySnapshot(visibility_slot_idx=np.array([], dtype=bool),
                                                                             visibility_time=TimeDelta(0, format='sec'))
                continue

            # Is the program excluded on a given night due to block scheduling
            can_schedule = nc[night_idx].filter.program_filter(prog)
            if not can_schedule:
                visibility_snapshots[str(int(jday.jd))] = VisibilitySnapshot(visibility_slot_idx=np.array([], dtype=bool),
                                                                             visibility_time=TimeDelta(0, format='sec'))
                continue

            # Calculate the time slot indices for the night where:
            # 1. The sun altitude requirement is met (precalculated in night_events)
            # 2. The sky background constraint is met
            # 3. The elevation constraints are met
            sa_idx = night_events.sun_alt_indices[night_idx]

            c_idx = np.where(
                np.logical_and(target_snapshot.sky_brightness[sa_idx] <= target_snapshot.target_sb,
                               np.logical_and(targ_prop[sa_idx] >= elev_min,
                               targ_prop[sa_idx] <= elev_max))
            )[0]

            # Apply timing window constraints.
            # We always have at least one timing window. If one was not given, the program length will be used.
            for tw in timing_windows:
                tw_idx = np.where(
                    np.logical_and(night_events.times[night_idx][sa_idx[c_idx]] >= tw[0],
                                   night_events.times[night_idx][sa_idx[c_idx]] <= tw[1])
                )[0]
                visibility_slot_idx = np.append(visibility_slot_idx, sa_idx[c_idx[tw_idx]])

            # It seems this is not needed.
            # # Create a visibility filter that has an entry for every time slot over the night,
            # # with 0 if the target is not visible and 1 if it is visible.
            # visibility_slot_filter = np.zeros(len(night_events.times[night_idx]))
            # visibility_slot_filter.put(visibility_slot_idx, 1.0)

            # TODO: Guide star availability for moving targets and parallactic angle modes.

            # Calculate the visibility time, the ongoing summed remaining visibility time, and
            # the remaining visibility fraction.
            # If the denominator for the visibility fraction is 0, use a value of 0.
            visibility_time = TimeDelta(len(visibility_slot_idx) * time_slot_length.to_value(u.s), format='sec')

            visibility_snapshot = VisibilitySnapshot(visibility_slot_idx=visibility_slot_idx,
                                                     visibility_time=visibility_time)
            # Pass to int to eliminate decimals and to string to keep the keys after deserialization.
            visibility_snapshots[str(int(jday.jd))] = visibility_snapshot

    return visibility_snapshots, target_snapshots


# From calculator.py
def get_target_visibility(obs: Observation, time_period: Time, semesters: FrozenSet[Semester],
                          tv: Dict[str, VisibilitySnapshot], vis_table: Optional[TargetVisibilityTable] = None):
    """Given a time period it calculates the target visibility for that period"""

    sem, = semesters  # This forces us to do plans to max one seme ster.

    rem_visibility_time = 0.0 * u.h
    rem_visibility_frac_numerator = obs.exec_time() - obs.total_used()

    target_visibilities: Dict[NightIndex, TargetVisibility] = {}
    # print(f'get_target_visibility: tv not None? {tv is not None}')

    for ridx, jday in enumerate(reversed(time_period)):
        # Convert to the actual time grid index.
        night_idx = NightIndex(len(time_period) - ridx - 1)
        day = str(int(jday.jd))
        visibility_snapshot = tv[day] if tv is not None else vis_table[day]

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


# From collector.py
def calculate_target_info(obs: Observation,
                          time_grid,
                          semesters,
                          night_events,
                          tv: Dict[str, VisibilitySnapshot] = None,
                          target_snapshots: Dict[NightIndex, VisibilitySnapshot] = None,
                          vis_table: Optional[TargetVisibilityTable] = None) -> TargetInfoNightIndexMap:
    """
    For a given site, calculate the information for a target for all the nights in
    the time grid and store this in the _target_information.

    Some of this information may be repetitive as, e.g. the RA and dec of a target should not
    depend on the site, so sites whose twilights overlap with have this information repeated.

    Finally, this method can calculate the total amount of time that, for the observation,
    the target is visible, and the visibility fraction for the target as a ratio of the amount of
    time remaining for the observation to the total visibility time for the target from a night through
    to the end of the period.
    """
    # Get the night events.
    if obs.site not in night_events:
        raise ValueError(f'Requested obs {obs.id.id} target info for site {obs.site}, which is not included.')

    target_vis = get_target_visibility(obs, time_grid, semesters, tv, vis_table)

    target_info: TargetInfoNightIndexMap = {}

    for i in range(len(time_grid)):
        night_idx = NightIndex(i)
        # if target_snapshots is None:
        #     target_snapshot = calculate_target_snapshot(night_idx,
        #                                                 obs,
        #                                                 target,
        #                                                 night_events[obs.site],
        #                                                 time_grid[night_idx],
        #                                                 time_slot_length,
        #                                                 for_vis_calc= False)
        # else:
        target_snapshot = target_snapshots[night_idx]
        ts = target_vis[night_idx]

        ti = TargetInfo(coord=target_snapshot.coord,
                        alt=target_snapshot.alt,
                        az=target_snapshot.az,
                        hourangle=target_snapshot.hourangle,
                        airmass=target_snapshot.airmass,
                        visibility_slot_idx=ts.visibility_slot_idx,
                        rem_visibility_frac=ts.rem_visibility_frac)

        target_info[NightIndex(night_idx)] = ti
    # Return all the target info for the base target in the Observation across the nights of interest.
    return target_info


def process_timing_windows(prog: Program, obs: Observation) -> List[Time]:
    """
    Given an Observation, convert the TimingWindow information in it to a simpler format
    to verify by converting each TimingWindow representation to a collection of Time frames
    based on the start, duration, repeat, and period.

    If no timing windows are given, then create one large timing window for the entire program.

    TODO: Look into simplifying to datetime instead of AstroPy Time.
    TODO: We may want to store this information in an Observation for future use.
    """
    if not obs.constraints or len(obs.constraints.timing_windows) == 0:
        # Create a timing window for the entirety of the program.
        windows = [Time([prog.start, prog.end])]
    else:
        windows = []
        for tw in obs.constraints.timing_windows:
            # The start time must be an astropy Time
            begin = tw.start
            duration = tw.duration.total_seconds() / 3600.0 * u.h
            repeat = max(1, tw.repeat)
            period = tw.period.total_seconds() / 3600.0 * u.h if tw.period is not None else 0.0 * u.h
            windows.extend([Time([begin + i * period, begin + i * period + duration]) for i in range(repeat)])

    return windows


def program_obs_vis(program_id, obs, program, time_grid, time_slot_length, semesters, night_configs,
                    night_events, vis_table):
    """Main routine for visibility calculations for observations in a program.
       This is run in difference processes by joblib"""

    base: Optional[Target] = obs.base_target()
    if base is None:
        _logger.error(f'Could not find base target for {obs.id.id}.')
        return None, None, None

    for_vis_calc = True if vis_table is None else False
    # print(f'program_obs_vis: for_vis_calc = {for_vis_calc}')

    # print(program.id.id, obs.id.id, night_events.keys())

    # Compute the timing window expansion for the observation.
    # Then, calculate the target information, which performs the visibility calculations.
    tw = process_timing_windows(program, obs)

    # Calculate visibilities
    vis_table_snap, target_snapshots = calculate_visibility(obs,
                                                            base,
                                                            program,
                                                            night_events[obs.site],
                                                            night_configs[obs.site],
                                                            time_grid,
                                                            tw,
                                                            time_slot_length,
                                                            for_vis_calc=for_vis_calc)

    ti = calculate_target_info(obs, time_grid, semesters, night_events, vis_table_snap, target_snapshots,
                               vis_table)

    return ti, base, vis_table_snap


class VisibilityCalculator(metaclass=Singleton):

    _SEMESTERS: ClassVar[FrozenSet[Semester]] = frozenset([Semester(2018, SemesterHalf('A')),
                                                          Semester(2018, SemesterHalf('B')),
                                                          Semester(2019, SemesterHalf('A')),
                                                          Semester(2019, SemesterHalf('B'))])

    def __init__(self):
        self.vis_table: Optional[TargetVisibilityTable] = None

    async def calculate(self) -> None:
        if config.collector.with_redis:
            all_semesters_vis_table = {}
            for semester in VisibilityCalculator._SEMESTERS:
                main_key = f"{semester}-{config.collector.time_slot_length}min"

                semester_vis_table = await redis_client.get_whole_dict(main_key)
                if semester_vis_table:
                    all_semesters_vis_table[semester] = semester_vis_table
                    _logger.info(f'Visibility calcs for {semester} from Redis retrieved.')

            self.vis_table = TargetVisibilityTable(all_semesters_vis_table)

        else:
            # fill the table manually. see fill_redis code.
            _logger.info("Visibility information will be calculated on runtime in the collector service.")

visibility_calculator = VisibilityCalculator()
