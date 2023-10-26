# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from datetime import datetime

from lucupy.minimodel.constraints import CloudCover, ImageQuality, Conditions, WaterVapor
from lucupy.minimodel.semester import SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.builder import ValidationBuilder
from scheduler.core.components.collector import *
from scheduler.core.eventsqueue.nightchanges import NightTimeline
from scheduler.core.output import print_plans
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.statscalculator import StatCalculator
from scheduler.core.eventsqueue import EveningTwilight, EventQueue, WeatherChange
from scheduler.services import logger_factory


if __name__ == '__main__':
    use_events = True

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
    sites = frozenset({Site.GS})
    num_nights_to_schedule = 1
    night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))

    queue = EventQueue(night_indices, sites)
    builder = ValidationBuilder(Sources(), queue)
    collector = builder.build_collector(
        start=start,
        end=end,
        sites=sites,
        semesters=frozenset([Semester(2018, SemesterHalf.B)]),
        blueprint=collector_blueprint
    )
    timeslot_length = collector.time_slot_length.to_datetime()

    # Add a twilight event for every night at each site.
    for site in sites:
        night_events = collector.get_night_events(site)
        for night_idx in night_indices:
            twilight_time = night_events.twilight_evening_12[night_idx].to_datetime()
            twilight = EveningTwilight(start=twilight_time, reason='Evening 12° Twilight', site=site)
            queue.add_event(night_idx, site, twilight)

    if use_events:
        # Create a weather event at GS that starts two hours after twilight on the first night of 2018-09-30,
        # which is why we look up the night events for night index 0 in calculating the time.
        night_events = collector.get_night_events(Site.GS)
        weather_change_time = night_events.twilight_evening_12[0].to_datetime() + timedelta(minutes=120)
        weather_change_south = WeatherChange(new_conditions=Conditions(iq=ImageQuality.IQ20,
                                                                       cc=CloudCover.CC50,
                                                                       sb=SkyBackground.SBANY,
                                                                       wv=WaterVapor.WVANY),
                                             start=weather_change_time,
                                             reason='IQ70 -> IQ20, CC70 -> CC50',
                                             site=Site.GS)
        queue.add_event(NightIndex(0), weather_change_south.site, weather_change_south)

    # Create the Collector and load the programs.
    collector.load_programs(program_provider_class=OcsProgramProvider,
                            data=programs)

    ValidationBuilder.update_collector(collector)  # ZeroTime observations

    # Initial weather for the selector.
    # We will reset this each night.
    initial_cc = CloudCover.CC50
    initial_iq = ImageQuality.IQ70

    selector = builder.build_selector(collector,
                                      num_nights_to_schedule=num_nights_to_schedule,
                                      default_cc=initial_cc,
                                      default_iq=initial_iq)

    # Prepare the optimizer.
    optimizer_blueprint = OptimizerBlueprint("GreedyMax")
    optimizer = builder.build_optimizer(blueprint=optimizer_blueprint)

    # Create the overall plans by night.
    overall_plans = {}
    night_timeline = NightTimeline({night_index: {site: [] for site in sites}
                                    for night_index in night_indices})

    for night_idx in sorted(night_indices):
        night_indices = np.array([night_idx])

        # Reset the Selector to the default weather for the night.
        # TODO: Make Selector accept site-specific values.
        selector.default_cc = initial_cc
        selector.default_iq = initial_iq

        # Run eventless timeline
        selection = selector.select(night_indices=night_indices)
        # Run the optimizer to get the plans for the first night in the selection.
        plans = optimizer.schedule(selection)

        # The starting twilight for the night for the site.
        night_start: Optional[datetime] = None

        for site in collector.sites:
            # Get the night events for the site: in this case, GS.
            # night_events = collector.get_night_events(site)
            # TODO: This needs to be a container that is sorted by start datetime of the events.
            # TODO: Right now, it is sorted, but only because we have added the events in datetime order.
            events_by_night = queue.get_night_events(night_idx, site)

            while events_by_night:
                event = events_by_night.popleft()
                match event:
                    case EveningTwilight(new_night_start, _, _):
                        if night_start is not None:
                            raise ValueError(f'Multiple twilight events for night index {night_idx} '
                                             f'at site {site.name}: was {night_start}, now {new_night_start}.')
                        night_start = new_night_start

                    case WeatherChange(_, _, _, new_conditions):
                        if night_start is None:
                            raise ValueError(f'Event for night index {night_idx} at site {site.name} occurred '
                                             f'before twilight: {event}.')
                        selector.default_iq = new_conditions.iq
                        selector.default_cc = new_conditions.cc

                    case _:
                        raise NotImplementedError(f'Received unsupported event: {event.__class__.__name__}')

                # Fetch a new selection given that the candidates and scores will need to be calculated based on
                # the event.
                event_start_time_slot = event.to_timeslot_idx(night_start, timeslot_length)
                selection = selector.select(night_indices=night_indices,
                                            sites=frozenset([event.site]),
                                            starting_time_slots={site: {night_idx: event_start_time_slot
                                                                        for night_idx in night_indices}})

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

    overall_plans = [p for _, p in sorted(overall_plans.items())]
    plan_summary = StatCalculator.calculate_plans_stats(overall_plans, collector)
    # print_plans(overall_plans)
    night_timeline.display()
    print('++++ FINAL PLANS ++++')
    print_plans(overall_plans)

    print('DONE')
