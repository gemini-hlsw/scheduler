# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import os
import time
from pathlib import Path

from astropy.time import Time
from lucupy.minimodel.site import ALL_SITES, Site
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties


from definitions import ROOT_DIR
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import SchedulerParameters, Engine
from scheduler.services import logger_factory
from scheduler.services.visibility import visibility_calculator

_logger = logger_factory.create_logger(__name__)

def main(*,
         programs_ids: Path = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.txt') -> None:

    # Set lucupy to Gemini
    ObservatoryProperties.set_properties(GeminiProperties)

    # Grab visibility calculations from Reddit
    asyncio.run(visibility_calculator.calculate())

    # Parsed program file (this replaces the program picker from Schedule)
    with open(programs_ids, 'r') as file:
        programs_list = [line.strip() for line in file if line.strip()[0] != '#']

    # Create Parameters
    # params = SchedulerParameters(start=Time("2025-09-30 08:00:00", format='iso', scale='utc'),
    #                              end=Time("2025-10-05 08:00:00", format='iso', scale='utc'),
    params=SchedulerParameters(start=Time("2026-01-20 08:00:00", format='iso', scale='utc'),
                                end=Time("2026-01-25 08:00:00", format='iso', scale='utc'),
                               sites=ALL_SITES,
                              #  sites=[Site.GN],
                              #  sites=[Site.GS],
                               mode=SchedulerModes.SIMULATION,
                               ranker_parameters=RankerParameters(vis_power=1.0, air_power=0.0),
                               semester_visibility=False,
                               num_nights_to_schedule=1,
                               programs_list=programs_list)
    engine = Engine(params)
    plan_summary, timelines = engine.schedule()
    # File output for future results comparison
    # outpath = os.path.join(os.environ['HOME'], 'gemini', 'sciops', 'softdevel', 'Queue_planning', 'sched_output')
    # timelines.display(output=os.path.join(outpath, 'dev_niri_s20181001_20250715.txt'))
    # Display to stdout
    timelines.display()
    print(plan_summary)

if __name__ == '__main__':
    t0 = time.time()
    main(programs_ids=Path(ROOT_DIR) / 'scheduler' / 'data' / 'gpp_program_ids.txt')
    print(f'Completed in {(time.time() - t0) / 60.} min')
