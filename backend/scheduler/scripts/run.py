# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from lucupy.minimodel.site import ALL_SITES, Site
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties


from definitions import ROOT_DIR
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import SchedulerParameters, Engine
from scheduler.services import logger_factory
from scheduler.services.sight.database.connection import init_db_engine

_logger = logger_factory.create_logger(__name__)

def main(*,
    programs_ids: Path = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.txt') -> None:

    # Set lucupy to Gemini
    ObservatoryProperties.set_properties(GeminiProperties)
    asyncio.run(init_db_engine())

    # Parsed program file (this replaces the program picker from Schedule)
    with open(programs_ids, 'r') as file:
        programs_list = [line.strip() for line in file if line.strip()[0] != '#']

    # Create Parameters
    params = SchedulerParameters(start=datetime.fromisoformat("2018-10-01 08:00:00").replace(tzinfo=ZoneInfo("UTC")),
                                 end=datetime.fromisoformat("2018-10-03 08:00:00").replace(tzinfo=ZoneInfo("UTC")),
                                 sites=ALL_SITES,
                                 mode=SchedulerModes.VALIDATION,
                                 ranker_parameters=RankerParameters(),
                                 semester_visibility=False,
                                 num_nights_to_schedule=1,
                                 programs_list=programs_list,
                                 use_local_visibility=True)
    engine = Engine(params)
    plan_summary, timelines = engine.schedule()
    timelines.display()

if __name__ == '__main__':
    main()