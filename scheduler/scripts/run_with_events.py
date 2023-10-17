# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from datetime import datetime
from math import ceil

from lucupy.minimodel.constraints import CloudCover, ImageQuality, Conditions, WaterVapor
from lucupy.minimodel.site import ALL_SITES
from lucupy.minimodel.semester import SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.builder import ValidationBuilder
from scheduler.core.components.collector import *
from scheduler.core.eventsqueue.events import Twilight
from scheduler.core.eventsqueue.nightchanges import NightTimeline
from scheduler.core.output import print_plans
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
    start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
    num_nights_to_schedule = int(round(end.jd - start.jd)) + 1

    queue = EventQueue([i for i in range(num_nights_to_schedule)], ALL_SITES)
    weather_change_south = WeatherChange(new_conditions=Conditions(iq=ImageQuality.IQANY,
                                                                   cc=CloudCover.CC50,
                                                                   sb=SkyBackground.SBANY,
                                                                   wv=WaterVapor.WVANY),
                                         start=datetime(2018, 10, 1, 10),
                                         reason='Worst image quality',
                                         site=Site.GS)

    queue.add_events(weather_change_south.site, [weather_change_south], 0, )

    builder = ValidationBuilder(Sources(), queue)
    # num_nights_to_schedule = 1
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
    night_timeline = NightTimeline({nidx: {site: [] for site in collector.sites} for nidx in range(total_nights)})

    for night_idx in range(selector.num_nights_to_schedule):
        night_indices = np.array([night_idx])

        # Run eventless timeline
        selection = selector.select(night_indices=night_indices)
        # Run the optimizer to get the plans for the first night in the selection.
        plans = optimizer.schedule(selection)

        for site in collector.sites:
            events_by_night = queue.get_night_events(night_idx, site)
            # Get the night events for the site: in this case, GS.
            night_events = collector.get_night_events(site)

            # The twilight evening time was calculated as a component of the night events.
            # We are only scheduling one day, so it is the only value in the array.
            twi_eve = night_events.twilight_evening_12[0]
            twi = Twilight(twi_eve, reason='Twilight', site=site)
            if site in plans[0].plans:
                night_timeline.add(night_idx=NightIndex(night_idx),
                                   site=site,
                                   time_slot=TimeslotIndex(0),
                                   event=twi,
                                   plan_generated=plans[0][site])

            if events_by_night:
                while events_by_night:
                    event = events_by_night.pop()
                    event_start_time_slot = ceil((event.start - start.to_datetime())/collector.time_slot_length.to_datetime())

                    if isinstance(event, WeatherChange):
                        selector.default_iq = event.new_conditions.iq
                        selector.default_cc = event.new_conditions.cc

                    selection = selector.select(night_indices=night_indices,
                                                sites=frozenset([event.site]),
                                                starting_time_slots={site: {night_idx: event_start_time_slot for night_idx in night_indices}})
                    # Run the optimizer to get the plans for the first night in the selection.
                    plans = optimizer.schedule(selection)
                    night_timeline.add(NightIndex(night_idx),
                                       site,
                                       TimeslotIndex(event_start_time_slot),
                                       event,
                                       plans[0][site])
                    collector.time_accounting(plans[0],
                                              sites=frozenset({site}),
                                              end_timeslot_bounds={site: TimeslotIndex(event_start_time_slot)})
        for site in collector.sites:
            plans[0][site] = night_timeline.get_final_plan(NightIndex(night_idx), site)

        overall_plans[night_idx] = plans[0]
        # Perform the time accounting on the plans.
        # collector.time_accounting(night_plans)
        selector.default_iq = ImageQuality.IQ70
        selector.default_cc = CloudCover.CC50
    overall_plans = [p for _, p in sorted(overall_plans.items())]
    plan_summary = StatCalculator.calculate_plans_stats(overall_plans, collector)
    # print_plans(overall_plans)
    night_timeline.display()
    print('++++ FINAL PLANS ++++')
    print_plans(overall_plans)

    print('DONE')
