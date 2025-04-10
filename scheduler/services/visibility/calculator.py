from dataclasses import dataclass
from typing import final, Dict, List, Any, Optional, ClassVar, FrozenSet, Generator

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

from scheduler.config import config
from scheduler.core.calculations import NightEvents
from scheduler.services.proper_motion import ProperMotionCalculator
from scheduler.services.ephemeris import EphemerisCalculator
from scheduler.services.redis_client import redis_client
from scheduler.services.logger_factory import create_logger
from scheduler.services.resource import NightConfiguration

from .snapshot import VisibilitySnapshot, TargetSnapshot
from scheduler.core.meta import Singleton

from scheduler.config import config

_logger = create_logger(__name__)

__all__ = [
    'calculate_target_snapshot',
    'visibility_calculator',
    'TargetVisibility',
    'TargetVisibilityTable'
]


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
            start_time_slot = time2slots(time_slot_length.to_datetime(), sunset_to_twi.to_datetime())
            end_time_slot = start_time_slot + num_time_slots

            # We must take every x minutes where x is the time slot length in minutes.
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


        return TargetSnapshot(max_dec=np.max(coord.dec),
                              min_dec=np.min(coord.dec),
                              alt=alt,
                              az=az,
                              par_ang=par_ang,
                              hourangle=hourangle,
                              airmass=airmass,
                              target_sb=targ_sb,
                              sky_brightness=sb)
    else:
        return TargetSnapshot(max_dec=np.max(coord.dec),
                              min_dec=np.min(coord.dec),
                              alt=alt,
                              az=az,
                              hourangle=hourangle,
                              airmass=airmass)


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


    def get_target_visibility(self, obs: Observation, time_period: Time, semesters: FrozenSet[Semester], tv: Dict[str, VisibilitySnapshot]):
        """Given a time period it calculates the target visibility for that period"""

        sem, = semesters  # This forces us to do plans to max one semester.

        rem_visibility_time = 0.0 * u.h
        rem_visibility_frac_numerator = obs.exec_time() - obs.total_used()

        target_visibilities: Dict[NightIndex, TargetVisibility] = {}

        for ridx, jday in enumerate(reversed(time_period)):
            # Convert to the actual time grid index.
            night_idx = NightIndex(len(time_period) - ridx - 1)
            day = str(int(jday.jd))
            visibility_snapshot = tv[day] if tv is not None else VisibilitySnapshot.from_dict(self.vis_table.get(sem, obs.id, day))

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

    @staticmethod
    def calculate_visibility(obs: Observation,
                             target: Target,
                             prog: Program,
                             night_events: NightEvents,
                             nc: dict[ndarray[Any, dtype[NightIndex]], NightConfiguration],
                             time_grid: Time,
                             timing_windows: List[Time],
                             time_slot_length: TimeDelta) -> Dict[str, VisibilitySnapshot]:
        """ Iterate over the time grid, checking to see if there is already a TargetInfo
                    for the target for the given day at the given site.
                    If so, we skip.
                    If not, we execute the calculations and store.
                    In order to properly calculate the:
                    * rem_visibility_time: total time a target is visible from the current night to the end of the period
                    * rem_visibility_frac: fraction of remaining observation length to rem_visibility_time
                    we want to process the nights BACKWARDS so that we can sum up the visibility time.
        """

        visibility_snapshots: Dict[str, VisibilitySnapshot] = {}

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

        return visibility_snapshots


visibility_calculator = VisibilityCalculator()
