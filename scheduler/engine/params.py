# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from datetime import datetime
from typing import final, Optional, FrozenSet

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
    start: Time
    end: Time
    sites: FrozenSet[Site]
    mode: SchedulerModes
    ranker_parameters: RankerParameters = field(default_factory=RankerParameters)
    semester_visibility: bool = True
    num_nights_to_schedule: Optional[int] = None
    program_file: Optional[str] = None

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
