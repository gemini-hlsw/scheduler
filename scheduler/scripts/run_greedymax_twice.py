# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

# NOTE: This is for testing purposes.
# An error was detected when trying to schedule a single site: Selector.select was still scoring
# observations for both sites. This tests the functionality to make sure that it now works correctly
# without having to go through any UI such as mercury.

import os
import logging

from lucupy.minimodel import ALL_SITES, CloudCover, ImageQuality, NightIndex, SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.builder import ValidationBuilder
from scheduler.core.components.collector import *
from scheduler.core.eventsqueue import EventQueue
from scheduler.core.output import print_collector_info, print_plans
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.services import logger_factory


if __name__ == '__main__':
    logger = logger_factory.create_logger(__name__, logging.INFO)
    ObservatoryProperties.set_properties(GeminiProperties)

    print('***** RUN 1: GN and GS *****')
    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD'],
        1.0
    )

    # start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    # end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
    start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-05 08:00:00", format='iso', scale='utc')
    num_nights_to_schedule = 5
    # num_nights_to_schedule = int(round(end.jd - start.jd)) + 1
    builder = ValidationBuilder(Sources(),
                                EventQueue([i for i in range(num_nights_to_schedule)], ALL_SITES))
    collector = builder.build_collector(
        start=start,
        end=end,
        sites=ALL_SITES,
        semesters=frozenset([Semester(2018, SemesterHalf.B)]),
        blueprint=collector_blueprint
    )
    collector.load_programs(program_provider_class=OcsProgramProvider,
                            data=programs)
    print_collector_info(collector, samples=60)
    ValidationBuilder.reset_collector_obseravtions(collector)

    cc = CloudCover.CC50
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
        night_indices = np.array([night_idx])
        selection = selector.select(night_indices=night_indices)

        # Run the optimizer to get the plans for the first night in the selection.
        plans = optimizer.schedule(selection)
        night_plans = plans[0]

        # Store the plans in the overall_plans array for that night.
        # TODO: This might be an issue. We may need to index nights (plans) in optimizer by night_idx.
        overall_plans[night_idx] = night_plans

        # Perform the time accounting on the plans.
        collector.time_accounting(night_plans)

    overall_plans = [p for _, p in sorted(overall_plans.items())]
    print_plans(overall_plans)

    print('\n\n\n***** RUN 2: GS ONLY *****')
    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD'],
        1.0
    )

    # start = Time("2018-10-04 08:00:00", format='iso', scale='utc')
    # end = Time("2018-10-07 08:00:00", format='iso', scale='utc')
    start = Time("2018-10-02 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-04 08:00:00", format='iso', scale='utc')
    num_nights_to_schedule = 3
    builder = ValidationBuilder(Sources(),
                                EventQueue([i for i in range(num_nights_to_schedule)], ALL_SITES))
    # num_nights_to_schedule = int(round(end.jd - start.jd)) + 1
    collector = builder.build_collector(
        start=start,
        end=end,
        sites=frozenset({Site.GS}),
        semesters=frozenset([Semester(2018, SemesterHalf.B)]),
        blueprint=collector_blueprint
    )
    collector.load_programs(program_provider_class=OcsProgramProvider,
                            data=programs)
    print_collector_info(collector, samples=60)
    ValidationBuilder.reset_collector_obseravtions(collector)

    cc = CloudCover.CC50
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
        night_indices = np.array([night_idx])
        selection = selector.select(night_indices=night_indices)

        # Run the optimizer to get the plans for the first night in the selection.
        plans = optimizer.schedule(selection)
        night_plans = plans[0]

        # Store the plans in the overall_plans array for that night.
        # TODO: This might be an issue. We may need to index nights (plans) in optimizer by night_idx.
        overall_plans[night_idx] = night_plans

        # Perform the time accounting on the plans.
        collector.time_accounting(night_plans)

    overall_plans = [p for _, p in sorted(overall_plans.items())]
    print_plans(overall_plans)

    print('DONE')
