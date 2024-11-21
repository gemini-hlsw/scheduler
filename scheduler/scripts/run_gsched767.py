# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

# For testing GSCHED-767 (Timing window parsing and FT note parsing)
# Bryan Miller

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, FrozenSet, Optional

import numpy as np
from astropy.time import Time
from lucupy.minimodel import NightIndex, TimeslotIndex, VariantSnapshot
from lucupy.minimodel.semester import Semester
from lucupy.minimodel.site import ALL_SITES, Site
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.timeutils import time2slots

from scheduler.core.builder.blueprint import CollectorBlueprint, SelectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.components.ranker import RankerParameters, DefaultRanker
from scheduler.core.components.changemonitor import ChangeMonitor, TimeCoordinateRecord
from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
from scheduler.core.output import print_collector_info, print_plans
from scheduler.core.plans import Plans
from scheduler.core.eventsqueue import EveningTwilightEvent, Event, EventQueue, MorningTwilightEvent, WeatherChangeEvent
from scheduler.core.sources.sources import Sources
from scheduler.core.statscalculator import StatCalculator
from scheduler.services import logger_factory
from scheduler.services.visibility import visibility_calculator
from scheduler.core.components.ranker import RankerParameters

_logger = logger_factory.create_logger(__name__)


def main(*,
         verbose: bool = False,
         start: Optional[Time] = Time("2018-10-01 08:00:00", format='iso', scale='utc'),
         end: Optional[Time] = Time("2018-10-03 08:00:00", format='iso', scale='utc'),
         sites: FrozenSet[Site] = ALL_SITES,
         ranker_parameters: RankerParameters = RankerParameters(),
         # semester_visibility: bool = False,
         semester_visibility: bool = True,
         # num_nights_to_schedule: Optional[int] = 1,
         num_nights_to_schedule: Optional[int] = None,
         programs_ids: Optional[str] = None) -> None:
    use_redis = True
    ObservatoryProperties.set_properties(GeminiProperties)
    asyncio.run(visibility_calculator.calculate())

    print(programs_ids)

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD', 'C'],
        1.0
    )

    semesters = frozenset([Semester.find_semester_from_date(start.datetime),
                           Semester.find_semester_from_date(end.datetime)])

    if semester_visibility:
        end_date = max(s.end_date() for s in semesters)
        end_vis = Time(datetime(end_date.year, end_date.month, end_date.day).strftime("%Y-%m-%d %H:%M:%S"))
        diff = end - start + 1
        diff = int(diff.jd)
        night_indices = frozenset(NightIndex(idx) for idx in range(diff))
        num_nights_to_schedule = diff
    else:
        night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))
        end_vis = end
        if not num_nights_to_schedule:
            raise ValueError("num_nights_to_schedule can't be None when visibility is given by end date")

    queue = EventQueue(night_indices, sites)
    builder = ValidationBuilder(Sources(), queue)

    # check if path exist and read
    f_programs = None
    if programs_ids:
        programs_path = Path(programs_ids)

        if programs_path.exists():
            f_programs = programs_path
        else:
            raise ValueError(f'Path {programs_path} does not exist.')

    # Create the Collector, load the programs, and zero out the time used by the observations.
    _logger.info("Creating collector")
    collector = builder.build_collector(
        start=start,
        end=end_vis,
        num_of_nights=num_nights_to_schedule,
        sites=sites,
        semesters=semesters,
        with_redis=use_redis,
        blueprint=collector_blueprint,
        program_list=f_programs
    )
    print('Collector done')

    # print_collector_info(collector)
    # print(collector.get_program_ids())

    for prog_id in collector.get_program_ids():
        p = collector.get_program(prog_id)
        print(f"{p.id.id} Start: {p.start}, End: {p.end}")
        for o in p.observations():
            print(f'\t{o.id.id} {o.constraints.timing_windows}')

    print('Done')

if __name__ == '__main__':
    main(ranker_parameters=RankerParameters(),
         # programs_ids=None,
         programs_ids="/Users/bmiller/gemini/sciops/softdevel/Queue_planning/program_ids_short.txt",
         verbose=False)
