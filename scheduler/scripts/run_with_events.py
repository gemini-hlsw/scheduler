# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import logging
from datetime import datetime

from lucupy.minimodel.constraints import CloudCover, ImageQuality, Conditions, WaterVapor
from lucupy.minimodel.site import ALL_SITES
from lucupy.minimodel.semester import SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.builder import ValidationBuilder
from scheduler.core.components.collector import *
from scheduler.core.eventsqueue.nightchanges import NightChanges
from scheduler.core.output import print_collector_info, print_plans
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.statscalculator import StatCalculator
from scheduler.core.eventsqueue import EventQueue, WeatherChange
from scheduler.services import logger_factory


if __name__ == '__main__':
    logger = logger_factory.create_logger(__name__, logging.INFO)
    ObservatoryProperties.set_properties(GeminiProperties)

    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD'],
        1.0
    )
    queue = EventQueue()
    weather_change = WeatherChange(new_conditions=Conditions(iq=ImageQuality.IQ20,
                                                             cc=CloudCover.CC50,
                                                             sb=SkyBackground.SBANY,
                                                             wv=WaterVapor.WVANY),
                                   start=datetime(2018, 10, 1, 10),
                                   reason='Worst image quality',
                                   )
    queue.add_events(weather_change, 0)

    builder = ValidationBuilder(Sources(), queue)

    start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
    num_nights_to_schedule = int(round(end.jd - start.jd)) + 1
    collector = builder.build_collector(
        start=start,
        end=end,
        sites=ALL_SITES,
        semesters=frozenset([Semester(2018, SemesterHalf.B)]),
        blueprint=collector_blueprint
    )
    # Create the Collector and load the programs.
    collector.load_programs(program_provider_class=OcsProgramProvider,
                            data=programs)

    ValidationBuilder.update_collector(collector)  # ZeroTime observations

    # TODO: SET THE WEATHER HERE.
    # CC values are CC50, CC70, CC85, CCANY. Default is CC50 if not passed to build_selector.
    cc = CloudCover.CC50

    # IQ values are IQ20, IQ70, IQ85, and IQANY. Default is IQ70 if not passed to build_selector.
    iq = ImageQuality.IQ70

    selector = builder.build_selector(collector,
                                      num_nights_to_schedule=num_nights_to_schedule,
                                      default_cc=cc,
                                      default_iq=iq)

    # Prepare the optimizer.
    optimizer_blueprint = OptimizerBlueprint(
        "GreedyMax"
    )
    optimizer = builder.build_optimizer(
        blueprint=optimizer_blueprint
    )

    # The total nights for which visibility calculations have been done.
    total_nights = len(collector.time_grid)

    # Create the overall plans by night.
    overall_plans = {}
    for night_idx in range(selector.num_nights_to_schedule):
        events_by_night = queue.get_night_events(night_idx)
        night_indices = np.array([night_idx])
        changes = NightChanges()
        if events_by_night:
            while events_by_night:
                event = events_by_night.pop()
                selection = selector.select(night_indices=night_indices)
                # Run the optimizer to get the plans for the first night in the selection.
                plans = optimizer.schedule(selection)
                if not changes.lookup:
                    changes.lookup[start] = plans[0]
                else:
                    changes.lookup[event.start] = plans[0]

                # TODO: Add logic for handle event behavior.
            # TODO: For now lets just display the final plan
            night_plans = changes.get_final_plans()
        else:
            # eventless run
            selection = selector.select(night_indices=night_indices)
            # Run the optimizer to get the plans for the first night in the selection.
            plans = optimizer.schedule(selection)
            night_plans = plans[0]

        overall_plans[night_idx] = night_plans

        # Perform the time accounting on the plans.
        collector.time_accounting(night_plans)

    overall_plans = [p for _, p in sorted(overall_plans.items())]
    plan_summary = StatCalculator.calculate_plans_stats(overall_plans, collector)
    print_plans(overall_plans)

    print('DONE')
