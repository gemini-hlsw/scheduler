# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import timedelta
from typing import Dict, FrozenSet, Optional

import numpy as np
from astropy.time import Time
from lucupy.minimodel import NightIndex, TimeslotIndex
from lucupy.minimodel.constraints import CloudCover, ImageQuality, VariantChange
from lucupy.minimodel.semester import Semester, SemesterHalf
from lucupy.minimodel.site import ALL_SITES, Site
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.core.builder.blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.components.ranker import RankerParameters, DefaultRanker
from scheduler.core.components.changemonitor import ChangeMonitor, TimeCoordinateRecord
from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
from scheduler.core.output import print_collector_info, print_plans
from scheduler.core.plans import Plans
from scheduler.core.eventsqueue import EveningTwilightEvent, Event, EventQueue, MorningTwilightEvent, WeatherChangeEvent
from scheduler.core.sources import Sources
from scheduler.services import logger_factory


# _logger = logger_factory.create_logger(__name__)
class _logger:
    @staticmethod
    def info(message: str):
        print(f'INFO: {message}')
    @staticmethod
    def error(message: str):
        print(f'ERROR: {message}')

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

    # Create the ChangeMonitor and keep track of when we should recalculate the plan for each site.
    change_monitor = ChangeMonitor(collector=collector, selector=selector)
    next_update: Dict[Site, Optional[TimeCoordinateRecord]] = {site: None for site in sites}

    # Add the twilight events for every night at each site.
    # The morning twilight will force time accounting to be done on the last generated plan for the night.
    for site in sorted(sites, key=lambda site: site.name):
        night_events = collector.get_night_events(site)
        for night_idx in night_indices:
            eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone).replace(tzinfo=None)
            eve_twi = EveningTwilightEvent(time=eve_twi_time, description='Evening 12° Twilight')
            queue.add_event(night_idx, site, eve_twi)

            # Add one time slot to the morning twilight to make sure time accounting is done for entire night.
            # morn_twilight_time = night_events.twilight_morning_12[night_idx].to_datetime(site.timezone)
            # morn_twilight = MorningTwilightEvent(time=morn_twilight_time, description='Morning 12° Twilight')
            # queue.add_event(night_idx, site, morn_twilight)
            morn_twi_time = night_events.twilight_morning_12[night_idx].to_datetime(site.timezone).replace(tzinfo=None)
            morn_twi = MorningTwilightEvent(time=morn_twi_time, description='Morning 12° Twilight')
            queue.add_event(night_idx, site, morn_twi)

    if test_events:
        # Create a weather event at GS that starts two hours after twilight on the first night of 2018-09-30,
        # which is why we look up the night events for night index 0 in calculating the time.
        night_events = collector.get_night_events(Site.GS)
        event_night_idx = 0
        site = Site.GS
        eve_twi_time = night_events.twilight_evening_12[event_night_idx].to_datetime(site.timezone).replace(tzinfo=None)
        weather_change_time = eve_twi_time + timedelta(minutes=120)
        weather_change_south = WeatherChangeEvent(time=weather_change_time,
                                                  description='IQ -> IQ20, CC -> CC50',
                                                  variant_change=VariantChange(iq=ImageQuality.IQ20,
                                                                               cc=CloudCover.CC50,
                                                                               wind_dir=None,
                                                                               wind_spd=None))
        # queue.add_event(NightIndex(0), Site.GS, weather_change_south)

    # Prepare the optimizer.
    optimizer_blueprint = OptimizerBlueprint("GreedyMax")
    optimizer = builder.build_optimizer(blueprint=optimizer_blueprint)

    # Create the overall plan by night. We will convert these into a List[Plans] at the end.
    overall_plans: Dict[NightIndex, Plans] = {}
    nightly_timeline = NightlyTimeline()

    for night_idx in sorted(night_indices):
        night_indices = np.array([night_idx])
        ranker = DefaultRanker(collector, night_indices, sites, params=ranker_parameters)

        # TODO: When weather service is working again, we will not do this.
        # Reset the Selector to the default weather for the night and reset the time record. The evening twilight
        # should trigger the initial plan generation.
        for site in sites:
            cc_value = cc_per_site and cc_per_site.get(site)
            iq_value = iq_per_site and iq_per_site.get(site)
            selector.update_cc_and_iq(site, cc_value, iq_value)

        # TODO: This needs reworking. We should be treating time linearly instead of iterating over sites and
        # TODO: processing them one after the other.
        for site in collector.sites:
            # Site name so we can change this if we see fit.
            site_name = site.name

            # Plan and event queue management.
            plans: Optional[Plans] = None
            events_by_night = queue.get_night_events(night_idx, site)
            if events_by_night.is_empty():
                raise RuntimeError(f'No events for site {site_name} for night {night_idx}.')

            # We need the start of the night for checking if an event has been reached.
            # Next update indicates when we will recalculate the plan.
            night_events = collector.get_night_events(site)
            night_start = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone).replace(tzinfo=None)
            next_update[site] = None

            # TODO: Do we want the timeslot counter in its own class? If we are going to make this work for GPP,
            # TODO: we will need one for real time and one for validation mode simulated time.
            # timeslot counter for the night for the site, and when to process the next event.
            current_timeslot: TimeslotIndex = TimeslotIndex(0)
            next_event: Optional[Event] = None
            next_event_timeslot: Optional[TimeslotIndex] = None
            night_done = False

            # If next_update[site] is None, we are just starting.
            # We continue until the next update to perform indicates that the night is done.
            # LOGIC:
            # 1. If there is no next event, then get one.
            # 2. If there is a next event, calculate its processing time.
            # 3. If there is no existing processing time, set
            # while next_update[site] is None or not next_update[site].done:
            while not night_done:
                # If our next update isn't done, and we are out of events, we're missing the morning twilight.
                if next_event is None and events_by_night.is_empty():
                    raise RuntimeError(f'No morning twilight found for site {site_name} for night {night_idx}.')

                # TODO: When do we do this and when do we set next_event to None?
                # IDEA: the next event can trigger an update before OR at the currently scheduled update.
                # In the case of something like a fault, it may trigger before.
                # In the case of weather, the weather will be changed, but it might still let the current observation
                #   finish, in which case, it will happen at (instead of) the current update.
                # I dont think it can happen AFTER the currently planned update. Think about this.
                # Example: for twilights, we want these to be scheduled.

                # Keep processing events until we find an event in the future, which will be stored in next_event.
                while next_event is None or current_timeslot >= next_event_timeslot:
                    if next_event is None:
                        # Get a new event and determine when it should be processed.
                        next_event = events_by_night.pop_next_event()
                        next_event_timeslot = next_event.to_timeslot_idx(night_start, time_slot_length)
                        _logger.info(f'Received event for site {site_name} for night idx {night_idx} to be processed '
                                     f'at timeslot {next_event_timeslot}: {next_event.__class__.__name__}')

                    # At this point, there should always be a next_event entry.
                    # If we have reached the next event, then process it.
                    print(f'Current timeslot: {current_timeslot}, next_event_timeslot: {next_event_timeslot}')
                    if current_timeslot >= next_event_timeslot:
                        print('Processing...')
                        if current_timeslot > next_event_timeslot:
                            _logger.error(f'Event was supposed to be processed at site {site.name} for night '
                                          f'{night_idx} at time {next_event_timeslot}, but now timeslot is '
                                          f'{current_timeslot}: {next_event.__class__.__name__}')

                        # Process the event to find out if we should recalculate the plan based on it and when.
                        time_record = change_monitor.process_event(site, next_event, plans)
                        if time_record is not None:
                            # In the case that:
                            # * there is no next update scheduled; or
                            # * this update happens before the next update
                            # then set to this update.
                            if next_update[site] is None or time_record.timeslot_idx < next_update[site].timeslot_idx:
                                next_update[site] = time_record

                        # As we processed the event, set it and the next_event_timeslot to None to indicate this
                        # and to continue processing to get next event.
                        next_event = None
                        next_event_timeslot = None

                # If we have reached the time for the next update, then perform it.
                # This is where we perform time accounting (if necessary), get a selection, and create a plan.
                if next_update[site] is not None and current_timeslot >= next_update[site].timeslot_idx:
                    # Remove the update and perform it.
                    update = next_update[site]
                    next_update[site] = None

                    if current_timeslot > update.timeslot_idx:
                        _logger.error(f'Plan update was supposed to happen at site {site.name} for night {night_idx} '
                                      f'at timeslot {update.timeslot_idx}, but now timeslot is {current_timeslot}.')

                    # We will update the plan up until the time that the update happens.
                    # If this update corresponds to the night being done, then use None.
                    if update.done:
                        end_timeslot_bounds = {}
                    else:
                        end_timeslot_bounds = {site: update.timeslot_idx}

                    # If there was an old plan and time accounting is to be done, then process it.
                    if plans is not None and update.perform_time_accounting:
                        if update.done:
                            ta_description = 'for rest of night.'
                        else:
                            ta_description = f'up to timeslot {update.timeslot_idx}.'
                        _logger.info(f'Performing time accounting at site {site_name} for night {night_idx} '
                                     + ta_description)
                        collector.time_accounting(plans,
                                                  sites=frozenset({site}),
                                                  end_timeslot_bounds=end_timeslot_bounds)

                    # Get a new selection and request a new plan if the night is not done.
                    if not update.done:
                        _logger.info(f'Retrieving selection for {site_name} for night {night_idx} '
                                     f'starting at time slot {current_timeslot}.')
                        selection = selector.select(night_indices=night_indices,
                                                    sites=frozenset([site]),
                                                    starting_time_slots={site: {night_idx: current_timeslot
                                                                                for night_idx in night_indices}},
                                                    ranker=ranker)

                        # Right now the optimizer generates List[Plans], a list of plans indexed by
                        # every night in the selection. We only want the first one, which corresponds
                        # to the current night index we are looping over.
                        _logger.info(f'Running optimizer for {site_name} for night {night_idx} '
                                     f'starting at time slot {current_timeslot}.')
                        plans = optimizer.schedule(selection)[0]
                        nightly_timeline.add(NightIndex(night_idx),
                                             site,
                                             current_timeslot,
                                             update.event,
                                             plans[site])

                    # If the update indicates that the night is done, then exit.
                    print(f'Setting night_done from {night_done} to {update.done}.')
                    night_done = update.done

                # Advance the current timeslot.
                current_timeslot += 1
                print(f'Timeslot now: {current_timeslot}')

        # Piece together the plans for the night to get the overall plans.
        # This is rather convoluted because of the confusing relationship between Plan, Plans, and NightlyTimeline.
        # TODO: There appears to be a bug here. See GSCHED-517.
        _logger.info(f'Assembling plans for night index {night_idx}.')
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
