# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime

from lucupy.minimodel.constraints import CloudCover, ImageQuality, Conditions, WaterVapor
from lucupy.minimodel.semester import SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.core.builder.blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.components.collector import *
from scheduler.core.components.ranker import RankerParameters, DefaultRanker
from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
from scheduler.core.output import print_collector_info, print_plans
from scheduler.core.eventsqueue import EveningTwilightEvent, EventQueue, MorningTwilightEvent, WeatherChangeEvent
from scheduler.services import logger_factory


logger = logger_factory.create_logger(__name__)


def main(*,
         verbose: bool = False,
         start: Optional[Time] = Time("2018-10-01 08:00:00", format='iso', scale='utc'),
         end: Optional[Time] = Time("2018-10-03 08:00:00", format='iso', scale='utc'),
         num_nights_to_schedule: int = 1,
         test_events: bool = False,
         sites: FrozenSet[Site] = ALL_SITES,
         ranker_parameters: RankerParameters = RankerParameters(),
         cc_per_site: Optional[Dict[Site, CloudCover]] = None,
         iq_per_site: Optional[Dict[Site, ImageQuality]] = None) -> None:
    ObservatoryProperties.set_properties(GeminiProperties)

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD'],
        1.0
    )
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
    time_slot_length = collector.time_slot_length.to_datetime()

    if verbose:
        print_collector_info(collector)

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
            eve_twilight = EveningTwilightEvent(time=eve_twilight_time, description='Evening 12° Twilight')
            queue.add_event(night_idx, site, eve_twilight)

            # Add one time slot to the morning twilight to make sure time accounting is done for entire night.
            morn_twilight_time = night_events.twilight_morning_12[night_idx].to_datetime()
            morn_twilight = MorningTwilightEvent(time=morn_twilight_time, description='Morning 12° Twilight')
            queue.add_event(night_idx, site, morn_twilight)

    if test_events:
        # Create a weather event at GS that starts two hours after twilight on the first night of 2018-09-30,
        # which is why we look up the night events for night index 0 in calculating the time.
        night_events = collector.get_night_events(Site.GS)
        event_night_idx = 0
        weather_change_time = night_events.twilight_evening_12[event_night_idx].to_datetime() + timedelta(minutes=120)
        weather_change_south = WeatherChangeEvent(time=weather_change_time,
                                                  description='IQ -> IQ20, CC -> CC50',
                                                  new_conditions=Conditions(iq=ImageQuality.IQ20,
                                                                            cc=CloudCover.CC50,
                                                                            sb=SkyBackground.SBANY,
                                                                            wv=WaterVapor.WVANY))
        queue.add_event(NightIndex(0), Site.GS, weather_change_south)

    # Prepare the optimizer.
    optimizer_blueprint = OptimizerBlueprint("GreedyMax")
    optimizer = builder.build_optimizer(blueprint=optimizer_blueprint)

    # Create the overall plan by night. We will convert these into a List[Plans] at the end.
    overall_plans: Dict[NightIndex, Plans] = {}
    nightly_timeline = NightlyTimeline()

    for night_idx in sorted(night_indices):
        night_indices = np.array([night_idx])

        ranker = DefaultRanker(collector, night_indices, sites, params=ranker_parameters)
        # Reset the Selector to the default weather for the night.
        for site in sites:
            cc_value = cc_per_site and cc_per_site.get(site)
            iq_value = iq_per_site and iq_per_site.get(site)
            selector.update_cc_and_iq(site, cc_value, iq_value)

        for site in collector.sites:
            # Initial values until the evening twilight is executed, and a plan is made.
            night_start: Optional[datetime] = None
            night_done = False
            plans = None

            # Get the night events for the site: in this case, GS.
            # night_events = collector.get_night_events(site)
            # TODO: This needs to be a container that is sorted by start datetime of the events.
            # TODO: Right now, it is sorted, but only because we have added the events in datetime order.
            events_by_night = queue.get_night_events(night_idx, site)

            while events_by_night.has_more_events():
                event = events_by_night.next_event()
                logger.info(f"Received event for night idx {night_idx} at site {site.name}: {event.__class__.__name__}")
                match event:
                    case EveningTwilightEvent(new_night_start, _):
                        if night_start is not None:
                            raise ValueError(f'Multiple evening twilight events for night index {night_idx} '
                                             f'at site {site.name}: was {night_start}, now {new_night_start}.')
                        night_start = new_night_start

                    case MorningTwilightEvent():
                        # This just marks the end of the observing night and triggers the time accounting.
                        if night_start is None:
                            raise ValueError(f'Morning twilight event for night index {night_idx} '
                                             f'at site {site.name} before evening twilight event.')
                        night_start = None
                        night_done = True

                    case WeatherChangeEvent(_, _, new_conditions):
                        if night_start is None:
                            raise ValueError(f'Event for night index {night_idx} at site {site.name} occurred '
                                             f'before twilight: {event}.')
                        selector.update_conditions(site, new_conditions)

                    case _:
                        raise NotImplementedError(f'Received unsupported event: {event.__class__.__name__}')

                # Calculate the time slot of the event. Note that if the night is done, it is None.
                if night_done:
                    event_start_time_slot = None
                    end_timeslot_bounds = None
                else:
                    event_start_time_slot = event.to_timeslot_idx(night_start, time_slot_length)
                    end_timeslot_bounds = {site: TimeslotIndex(event_start_time_slot)}

                # If tbe following conditions are met:
                # 1. there are plans (i.e. plans have been generated by this point, meaning at least the
                #    evening twilight event has occurred); and
                # 2. a new plan is to be produced (TODO: GSCHED-515)
                # then perform time accounting.
                #
                # This will also perform the final time accounting when the night is done and the morning twilight
                # event has occurred.
                if plans is not None:
                    logger.info(f'Performing time accounting for night index {night_idx} '
                                f'at {site.name} up to timeslot {event_start_time_slot}.')
                    collector.time_accounting(plans,
                                              sites=frozenset({site}),
                                              end_timeslot_bounds=end_timeslot_bounds)

                # If the following conditions are met:
                # 1. the night is not done;
                # 2. a new plan is to be produced (TODO: GSCHED-515)
                # fetch a new selection and produce a new plan.
                if not night_done:
                    logger.info(f'Retrieving selection for night index {night_idx} '
                                f'at {site.name} starting at time slot {event_start_time_slot}.')
                    selection = selector.select(night_indices=night_indices,
                                                sites=frozenset([site]),
                                                starting_time_slots={site: {night_idx: event_start_time_slot
                                                                            for night_idx in night_indices}},
                                                ranker=ranker)

                    # Right now the optimizer generates List[Plans], a list of plans indexed by
                    # every night in the selection. We only want the first one, which corresponds
                    # to the current night index we are looping over.
                    logger.info(f'Running optimizer for night index {night_idx} '
                                f'at {site.name} starting at time slot {event_start_time_slot}.')
                    plans = optimizer.schedule(selection)[0]
                    nightly_timeline.add(NightIndex(night_idx),
                                         site,
                                         TimeslotIndex(event_start_time_slot),
                                         event,
                                         plans[site])

        # Piece together the plans for the night to get the overall plans.
        # This is rather convoluted because of the confusing relationship between Plan, Plans, and NightlyTimeline.
        # TODO: There appears to be a bug here. See GSCHED-517.
        logger.info(f'Assembling plans for night index {night_idx}.')
        night_events = {site: collector.get_night_events(site) for site in collector.sites}
        final_plans = Plans(night_events, NightIndex(night_idx))
        for site in collector.sites:
            final_plans[site] = nightly_timeline.get_final_plan(NightIndex(night_idx), site)
        overall_plans[night_idx] = final_plans

    # Make sure we have a list of the final plans sorted by night index.
    plans_list = [p for _, p in sorted(overall_plans.items())]
    # plan_summary = StatCalculator.calculate_plans_stats(overall_plans, collector)
    nightly_timeline.display()

    print('++++ FINAL PLANS ++++')
    print_plans(plans_list)  # List[Plans]
    print('DONE')
