# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from datetime import datetime
from typing import final, Optional, FrozenSet, List

from astropy.time import Time
from lucupy.minimodel import Site, ALL_SITES, Semester, NightIndex

from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters


__all__ = [
    'SchedulerParameters'
]


@final
@dataclass
class SchedulerParameters:
    """
    Initial parameters to start the scheduler engine.
    The start parameter represents both the initial local night that is going to be
    schedule and the initial date for the visibility calculation.

    Attributes:
        start (Time): start local date.
        end (Time): end local date. Represents the end of the calculated visibility period.
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
           params = SchedulerParameters(start=Time("2018-10-01 08:00:00", format='iso', scale='utc'),
                                         end=Time("2018-10-03 08:00:00", format='iso', scale='utc'),
                                         sites=ALL_SITES,
                                         mode=SchedulerModes.VALIDATION,
                                         ranker_parameters=RankerParameters(),
                                         semester_visibility=False,
                                         num_nights_to_schedule=1,
                                         programs_list=programs_list)
        ```
    """
    start: Time
    end: Time
    sites: FrozenSet[Site]
    mode: SchedulerModes
    ranker_parameters: RankerParameters = field(default_factory=RankerParameters)
    semester_visibility: bool = True
    num_nights_to_schedule: Optional[int] = None
    programs_list: Optional[List[str]] = None

    def __post_init__(self):
        self.semesters = frozenset([Semester.find_semester_from_date(self.start.datetime),
                                    Semester.find_semester_from_date(self.end.datetime)])

        if self.semester_visibility:
            end_date = max(s.end_date() for s in self.semesters)
            self.end_vis = Time(datetime(end_date.year, end_date.month, end_date.day).strftime("%Y-%m-%d %H:%M:%S"))
            diff = self.end - self.start + 1
            diff = int(diff.jd)

            self.num_nights_to_schedule = diff
            self.night_indices = frozenset(NightIndex(idx) for idx in range(diff))
        else:
            self.night_indices = frozenset(NightIndex(idx) for idx in range(self.num_nights_to_schedule))
            self.end_vis = self.end
            if not self.num_nights_to_schedule:
                raise ValueError("num_nights_to_schedule can't be None when visibility is given by end date")

    @staticmethod
    def from_json(received_params: dict) -> 'SchedulerParameters':
        return SchedulerParameters(Time(received_params['startTime'], format='iso', scale='utc'),
                                   Time(received_params['endTime'], format='iso', scale='utc'),
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
