# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import final, Optional, FrozenSet, List, NamedTuple, Dict

from astropy.time import Time
from lucupy.minimodel import Site, ALL_SITES, Semester, NightIndex
from pydantic import BaseModel

# from pydantic import BaseModel

from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters


__all__ = [
    'SchedulerParameters',
    'build_params_store',
    'BuildParameters'
]


@final
@dataclass
class SchedulerParameters:
    """
    Initial parameters to start the scheduler engine.
    The start parameter represents both the initial local night that is going to be
    schedule and the initial date for the visibility calculation.

    Attributes:
        start (Time): start UT date.
        end (Time): end UT date. Represents the end of the calculated visibility period.
        mode (SchedulerModes): The mode the Scheduler can be executed in. VALIDATION, SIMULATION or OPERATION.
        ranker_parameters (RankerParameters): Parameters that can be toggled in the Ranker to modify score.
        semester_visibility (bool): Overrides the end parameters and extends the visibility period to the end
            of the selected semester in start.
        num_nights_to_schedule (int, optional): Number of nights to schedule. Can't be bigger then the amount of nights
            in the visibility period. Defaults to None.
        programs_list (List[str], optional):  A list of ProgramID that allows a specific selection of programs to run.
            Defaults to None. If None, the default programs list in scheduler/data would be used.

    Examples:
        ```python

           from scheduler.engine import SchedulerParameters, Engine
           params = SchedulerParameters(start=datetime.fromisoformat("2018-10-01 08:00:00"),
                                         end=datetime.fromisoformat("2018-10-03 08:00:00"),
                                         sites=ALL_SITES,
                                         mode=SchedulerModes.VALIDATION,
                                         ranker_parameters=RankerParameters(),
                                         semester_visibility=False,
                                         num_nights_to_schedule=1,
                                         programs_list=programs_list)
        ```
    """
    start: datetime
    end: datetime = None
    sites: FrozenSet[Site] = ALL_SITES
    mode: SchedulerModes = SchedulerModes.OPERATION
    ranker_parameters: RankerParameters = field(default_factory=RankerParameters)
    semester_visibility: bool = True
    num_nights_to_schedule: Optional[int] = None
    programs_list: Optional[List[str]] = None

    def __post_init__(self):
        if self.end is not None and self.end > self.start:
            # The semester methods work on local dates, so have to subtract 1 day from UT dates
            self.semesters = frozenset([Semester.find_semester_from_date(self.start - timedelta(days=1)),
                                        Semester.find_semester_from_date(self.end - timedelta(days=1))])
        else:
            self.semesters = frozenset([Semester.find_semester_from_date(self.start) - timedelta(days=1)])

        if self.semester_visibility:
            end_date = max(s.end_date() for s in self.semesters)
            # end_date is a local date, so add 1 for UT
            end_date += timedelta(days=1)
            ut_hr = self.start.hour
            self.end_vis = datetime(end_date.year, end_date.month, end_date.day, hour=ut_hr, tzinfo=ZoneInfo("UTC"))
            if self.end is None:
                diff = 1
            else:
                diff = (self.end - self.start).days + 1

            self.num_nights_to_schedule = diff
            self.night_indices = frozenset(NightIndex(idx) for idx in range(diff))
        else:
            if not self.num_nights_to_schedule:
                raise ValueError("num_nights_to_schedule can't be None when visibility is given by end date")
            self.night_indices = frozenset(NightIndex(idx) for idx in range(self.num_nights_to_schedule))
            self.end_vis = self.end


    @staticmethod
    def from_json(received_params: dict) -> 'SchedulerParameters':
        return SchedulerParameters(datetime.fromisoformat(received_params['startTime']),
                                   datetime.fromisoformat(received_params['endTime']),
                                   frozenset([Site[received_params['sites'][0]]]) if len(received_params['sites']) < 2 else ALL_SITES,
                                   SchedulerModes[received_params['schedulerMode']],
                                   RankerParameters(thesis_factor=float(received_params['rankerParameters']['thesisFactor']),
                                                    power=int(received_params['rankerParameters']['power']),
                                                    wha_power=float(received_params['rankerParameters']['whaPower']),
                                                    met_power=float(received_params['rankerParameters']['metPower']),
                                                    vis_power=float(received_params['rankerParameters']['visPower'])),
                                   received_params['semesterVisibility'],
                                   received_params['numNightsToSchedule'],
                                   None)

    def __str__(self) -> str:
        return "Scheduler Parameters:\n" + \
            f"├─start: {self.start}\n" + \
            f"├─end: {self.end}\n" + \
            f"├─sites: {', '.join([site.name for site in self.sites])}\n" + \
            f"├─mode: {self.mode}\n" + \
            f"├─semester_visibility: {self.semester_visibility}\n" + \
            f"├─num_nights_to_schedule: {self.num_nights_to_schedule}\n" + \
            f"└─ranker_parameters: {self.ranker_parameters}"


class NightTimes(BaseModel):
    night_start: datetime | None = None
    night_end: datetime | None = None

    def start_time(self) -> Time:
        return Time(self.night_start, scale="utc")

    def end_time(self) -> Time:
        return Time(self.night_end, scale="utc")


class BuildParameters(BaseModel):
    """
    Specific parameters used to modify behavior on components on the SCP.

    night_start (datetime): Modify the start of the night, instead of evening twilight use this date.
    night_end (datetime): Modify the end of the night, instead of morning twilight use this date.
    """
    night_times: Dict[Site, NightTimes] | None = None
    visibility_start: datetime | None = None
    visibility_end: datetime | None = None
    program_list: List[str] | None = None

    def get_night_times(self):
        if self.night_times is None:
            return {}
        return {site: (
                    self.night_times[site].start_time(),
                    self.night_times[site].end_time()
                )
            for site, nt in self.night_times.items() if nt is not None
        }


class BuildParamsStore:
    """

    """
    def __init__(self) -> None:
        self._params = BuildParameters()

        self._lock = threading.Lock()

    def get(self) -> BuildParameters:
        with self._lock:
            return self._params

    def set(self,params: BuildParameters) -> None:
        with self._lock:
            self._params = params

build_params_store = BuildParamsStore()
