import json
from dataclasses import dataclass, field
from typing import final, Optional, FrozenSet

from astropy.time import Time
from lucupy.minimodel import Site, ALL_SITES

from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters


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

    @staticmethod
    def from_json(received_params: dict) -> 'SchedulerParameters':
        return SchedulerParameters(Time(received_params['startTime'], format='iso', scale='utc'),
                                   Time(received_params['endTime'], format='iso', scale='utc'),
                                   frozenset([Site[received_params['sites'][0]]]) if len(received_params) < 2 else ALL_SITES,
                                   SchedulerModes[received_params['schedulerMode']],
                                   RankerParameters(thesis_factor=float(received_params['rankerParameters']['thesisFactor']),
                                                    power=int(received_params['rankerParameters']['power']),
                                                    wha_power=float(received_params['rankerParameters']['whaPower']),
                                                    met_power=float(received_params['rankerParameters']['metPower']),
                                                    vis_power=float(received_params['rankerParameters']['visPower'])),
                                   received_params['semesterVisibility'],
                                   received_params['numNightsToSchedule'],
                                   None)
