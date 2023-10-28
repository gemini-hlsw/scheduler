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
from scheduler.core.eventsqueue import EveningTwilight, EventQueue, MorningTwilight, WeatherChange


def main(*,
         num_nights_to_schedule: int = 1,
         test_events: bool = False,
         sites: FrozenSet[Site] = ALL_SITES,
         cc_per_site: Optional[Dict[Site, CloudCover]] = None,
         iq_per_site: Optional[Dict[Site, ImageQuality]] = None) -> None:
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
    night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))

    queue = EventQueue(night_indices, sites)
    builder = ValidationBuilder(Sources(), queue)

    # Create the Collector, load the programs, and zero out the time used by the observations.
    collector = builder.build_collector(
        start=start,
        end=end,
        sites=sites,
        semesters=frozenset([Semester(2018, SemesterHalf.B)]),
        blueprint=collector_blueprint
    )
    collector.load_programs(program_provider_class=OcsProgramProvider,
                            data=programs)
    ValidationBuilder.reset_collector_obseravtions(collector)

    # Create the Selector.
    selector = builder.build_selector(collector,
                                      num_nights_to_schedule=num_nights_to_schedule,
                                      cc_per_site=cc_per_site,
                                      iq_per_site=iq_per_site)

    # Add the twilight events for every night at each site.
    # The morning twilight will force time accounting to be done on the last generated plan for the night.
    for site in sites:
        night_events = collector.get_night_events(site)
        for night_idx in night_indices:
            eve_twilight_time = night_events.twilight_evening_12[night_idx].to_datetime()
            eve_twilight = EveningTwilight(start=eve_twilight_time, reason='Evening 12° Twilight', site=site)
            queue.add_event(night_idx, site, eve_twilight)

            # Add one time slot to the morning twilight to make sure time accounting is done for entire night.
            morn_twilight_time = night_events.twilight_morning_12[night_idx].to_datetime()
            morn_twilight = MorningTwilight(start=morn_twilight_time, reason='Morning 12° Twilight', site=site)
            queue.add_event(night_idx, site, morn_twilight)

    if test_events:
        # Create a weather event at GS that starts two hours after twilight on the first night of 2018-09-30,
        # which is why we look up the night events for night index 0 in calculating the time.
        night_events = collector.get_night_events(Site.GS)
        event_night_idx = 0
        weather_change_time = night_events.twilight_evening_12[event_night_idx].to_datetime() + timedelta(minutes=120)
        weather_change_south = WeatherChange(new_conditions=Conditions(iq=ImageQuality.IQ20,
                                                                       cc=CloudCover.CC50,
                                                                       sb=SkyBackground.SBANY,
                                                                       wv=WaterVapor.WVANY),
                                             start=weather_change_time,
                                             reason='IQ -> IQ20, CC -> CC50',
                                             site=Site.GS)
        queue.add_event(NightIndex(0), weather_change_south.site, weather_change_south)

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
        for site in sites:
            cc_value = cc_per_site and cc_per_site.get(site)
            iq_value = iq_per_site and iq_per_site.get(site)
            selector.update_cc_and_iq(site, cc_value, iq_value)

        # Run eventless timeline
        selection = selector.select(night_indices=night_indices)
        # Run the optimizer to get the plans for the first night in the selection.
        plans = optimizer.schedule(selection)

        for site in collector.sites:
            # The starting twilight for the night for the site.
            night_start: Optional[datetime] = None
            night_done = False

            # Get the night events for the site: in this case, GS.
            # night_events = collector.get_night_events(site)
            # TODO: This needs to be a container that is sorted by start datetime of the events.
            # TODO: Right now, it is sorted, but only because we have added the events in datetime order.
            events_by_night = queue.get_night_events(night_idx, site)

            while events_by_night.has_more_events():
                event = events_by_night.next_event()
                match event:
                    case EveningTwilight(new_night_start, _, _):
                        if night_start is not None:
                            raise ValueError(f'Multiple evening twilight events for night index {night_idx} '
                                             f'at site {site.name}: was {night_start}, now {new_night_start}.')
                        night_start = new_night_start

                    case MorningTwilight():
                        # This just marks the end of the observing night and triggers the time accounting.
                        if night_start is None:
                            raise ValueError(f'Morning twilight event for night index {night_idx} '
                                             f'at site {site.name} before evening twilight event.')
                        night_start = None
                        night_done = True

                    case WeatherChange(_, _, affected_site, new_conditions):
                        if night_start is None:
                            raise ValueError(f'Event for night index {night_idx} at site {site.name} occurred '
                                             f'before twilight: {event}.')
                        selector.update_conditions(affected_site, new_conditions)

                    case _:
                        raise NotImplementedError(f'Received unsupported event: {event.__class__.__name__}')

                # If the night is not done, fetch a new selection given that the candidates and scores will need to be
                # calculated based on the event.
                event_start_time_slot = None
                if not night_done:
                    event_start_time_slot = event.to_timeslot_idx(night_start, collector.time_slot_length.to_datetime())
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

                # Perform the time accounting. If the night is done, we execute it to completion.
                end_timeslot_bounds = None if night_done else {site: TimeslotIndex(event_start_time_slot)}
                collector.time_accounting(plans[0],
                                          sites=frozenset({site}),
                                          end_timeslot_bounds=end_timeslot_bounds)

        # Piece together the plans for the night to get the overall plans.
        for site in collector.sites:
            plans[0][site] = night_timeline.get_final_plan(NightIndex(night_idx), site)
        overall_plans[night_idx] = plans[0]

    overall_plans = [p for _, p in sorted(overall_plans.items())]
    plan_summary = StatCalculator.calculate_plans_stats(overall_plans, collector)
    # print_plans(overall_plans)
    night_timeline.display()
    print('++++ FINAL PLANS ++++')
    print_plans(overall_plans)

    print('DONE')


if __name__ == '__main__':
    use_events = True
    main(test_events=True,
         num_nights_to_schedule=3,
         cc_per_site={Site.GS: CloudCover.CC70},
         iq_per_site={Site.GS: ImageQuality.IQ70})
