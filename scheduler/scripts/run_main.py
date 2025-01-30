# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
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


_logger = logger_factory.create_logger(__name__)


def main(*,
         verbose: bool = False,
         start: Optional[Time] = Time("2018-10-01 08:00:00", format='iso', scale='utc'),
         end: Optional[Time] = Time("2018-10-03 08:00:00", format='iso', scale='utc'),
         sites: FrozenSet[Site] = ALL_SITES,
         ranker_parameters: RankerParameters = RankerParameters(),
         # semester_visibility: bool = False,
         semester_visibility: bool = True,
         # num_nights_to_schedule: Optional[int] = 3,
         num_nights_to_schedule: Optional[int] = None,
         programs_ids: Optional[str] = None) -> None:
    ObservatoryProperties.set_properties(GeminiProperties)
    asyncio.run(visibility_calculator.calculate())


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
        with_redis=False,
        blueprint=collector_blueprint,
        program_list=f_programs
    )
    if verbose:
        print_collector_info(collector)

    # Create the Selector.
    _logger.info("Creating selector")
    selector_blueprint = SelectorBlueprint(
        'FLAT_MINUTES',
        30
    )
    selector = builder.build_selector(collector=collector,
                                      num_nights_to_schedule=num_nights_to_schedule,
                                      blueprint=selector_blueprint)

    # Create the ChangeMonitor and keep track of when we should recalculate the plan for each site.
    change_monitor = ChangeMonitor(collector=collector, selector=selector)

    # Don't use this now, but we will use it when scheduling sites at the same time.
    next_update: Dict[Site, Optional[TimeCoordinateRecord]] = {site: None for site in sites}

    # Prepare the optimizer.
    _logger.info("Creating optimizer")
    optimizer_blueprint = OptimizerBlueprint("GreedyMax")
    optimizer = builder.build_optimizer(blueprint=optimizer_blueprint)

    # Create the overall plan by night. We will convert these into a List[Plans] at the end.
    overall_plans: Dict[NightIndex, Plans] = {}

    time_slot_length = collector.time_slot_length.to_datetime()
    nightly_timeline = NightlyTimeline()

    # Initial weather conditions for a night.
    # These can occur if a weather reading is taken from timeslot 0 or earlier on a night.
    initial_variants: Dict[Site, Dict[NightIndex, Optional[VariantSnapshot]]] = \
        {site: {night_idx: None for night_idx in night_indices} for site in sites}

    # Add events for every site for each night.
    # The morning twilight will force time accounting to be done on the last generated plan for the night.
    for site in sites:
        night_events = collector.get_night_events(site)
        for night_idx in night_indices:
            eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
            eve_twi = EveningTwilightEvent(site=site, time=eve_twi_time, description='Evening 12° Twilight')
            queue.add_event(night_idx, site, eve_twi)

            # Get the closure events for the site for the given night data.
            night_date = eve_twi_time.date()
            morn_twi_time = night_events.twilight_morning_12[night_idx].to_datetime(site.timezone) - time_slot_length
            # morn_twi_slot = time2slots(time_slot_length, morn_twi_time - eve_twi_time)
            morn_twi_slot = night_events.num_timeslots_per_night[night_idx]

            # Get initial conditions for the nights
            initial_variants[site][night_idx] = collector.sources.origin.env.get_initial_conditions(site, night_date)

            # Get the weather events for the site for the given night date.
            # Get the VariantSnapshots for the times of the night where the variant changes.
            variant_changes_dict = collector.sources.origin.env.get_variant_changes_for_night(site, night_date)
            for variant_datetime, variant_snapshot in variant_changes_dict.items():
                variant_timeslot = time2slots(time_slot_length, variant_datetime - eve_twi_time)

                # If the variant happens before or at the first time slot, we set the initial variant for the night.
                # The closer to the first time slot, the more accurate, and the ordering on them will overwrite
                # the previous values.
                if variant_timeslot <= 0:
                    continue

                if variant_timeslot >= morn_twi_slot:
                    _logger.debug(f'WeatherChange for site {site.name}, night {night_idx}, occurs after '
                                  f'{morn_twi_slot}: ignoring.')
                    continue

                variant_datetime_str = variant_datetime.strftime('%Y-%m-%d %H:%M')
                weather_change_description = (f'Weather change at {site.name}, {variant_datetime_str}: '
                                              f'IQ -> {variant_snapshot.iq.name}, '
                                              f'CC -> {variant_snapshot.cc.name}')
                weather_change_event = WeatherChangeEvent(site=site,
                                                          time=variant_datetime,
                                                          description=weather_change_description,
                                                          variant_change=variant_snapshot)
                queue.add_event(night_idx, site, weather_change_event)

            # Process the unexpected closures for the night at the site.
            closure_set = collector.sources.origin.resource.get_unexpected_closures(site, night_date)
            for closure in closure_set:
                closure_start, closure_end = closure.to_events()
                queue.add_event(night_idx, site, closure_start)
                queue.add_event(night_idx, site, closure_end)

            # Process the fault reports for the night at the site.
            faults_set = collector.sources.origin.resource.get_faults(site, night_date)
            for fault in faults_set:
                fault_start, fault_end = fault.to_events()
                queue.add_event(night_idx, site, fault_start)
                queue.add_event(night_idx, site, fault_end)

            morn_twi = MorningTwilightEvent(site=site, time=morn_twi_time, description='Morning 12° Twilight')
            queue.add_event(night_idx, site, morn_twi)

    for night_idx in sorted(night_indices):
        night_indices = np.array([night_idx])
        ranker = DefaultRanker(collector, night_indices, sites, params=ranker_parameters)

        # TODO: This needs reworking. We should be treating time linearly instead of iterating over sites and
        # TODO: processing them one after the other.
        for site in sorted(sites, key=lambda site: site.name):
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
            night_start = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
            next_update[site] = None

            current_timeslot: TimeslotIndex = TimeslotIndex(0)
            next_event: Optional[Event] = None
            next_event_timeslot: Optional[TimeslotIndex] = None
            night_done = False

            # Set the initial variant for the site for the night. This may have been set above by weather
            # information obtained before or at the start of the night, and if not, then the lookup will give None,
            # which will reset to the default values as defined in the Selector.
            _logger.debug(f'Resetting {site_name} weather to initial values for night...')
            selector.update_site_variant(site, initial_variants[site][night_idx])

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
                # next_event should be the top of the event queue, the next event scheduled for the future.

                # There may be more than one event scheduled at the same time, so we must process all of them
                # that occur now and determine the one that causes the earliest recalculation of the plan.

                # If we don't know when the next event is, or we do and it is now, go over events.
                # Do we have events to perform? If so, consume all the events for the current time.
                if next_event_timeslot is None or current_timeslot >= next_event_timeslot:
                    # Stop if there are no more events.
                    while events_by_night.has_more_events():
                        top_event = events_by_night.top_event()
                        top_event_timeslot = top_event.to_timeslot_idx(night_start, time_slot_length)

                        # TODO: Check this over to make sure if there is an event now, it is processed.
                        # If we don't know the next event timeslot, set it.
                        if next_event_timeslot is None:
                            next_event_timeslot = top_event_timeslot
                            next_event = top_event

                        if current_timeslot > next_event_timeslot:
                            _logger.warning(f'Received event for {site_name} for night idx {night_idx} at timeslot '
                                            f'{next_event_timeslot} < current time slot {current_timeslot}.')

                        # The next event happens in the future, so record that time.
                        if top_event_timeslot > current_timeslot:
                            next_event_timeslot = top_event_timeslot
                            break

                        # We have an event that occurs at this time slot and is in top_event, so pop it from the
                        # queue and process it.
                        events_by_night.pop_next_event()
                        _logger.debug(
                            f'Received event for site {site_name} for night idx {night_idx} to be processed '
                            f'at timeslot {next_event_timeslot}: {next_event.__class__.__name__}')

                        # Process the event: find out when it should occur.
                        # If there is no next update planned, then take it to be the next update.
                        # If there is a next update planned, then take it if it happens before the next update.
                        # Process the event to find out if we should recalculate the plan based on it and when.
                        time_record = change_monitor.process_event(site, top_event, plans, night_idx)

                        if time_record is not None:
                            # In the case that:
                            # * there is no next update scheduled; or
                            # * this update happens before the next update
                            # then set to this update.
                            if next_update[site] is None or time_record.timeslot_idx < next_update[site].timeslot_idx:
                                next_update[site] = time_record
                                _logger.debug(f'Next update for site {site_name} scheduled at '
                                              f'timeslot {next_update[site].timeslot_idx}')

                # If there is a next update, and we have reached its time, then perform it.
                # This is where we perform time accounting (if necessary), get a selection, and create a plan.
                if next_update[site] is not None and current_timeslot >= next_update[site].timeslot_idx:
                    # Remove the update and perform it.
                    update = next_update[site]
                    next_update[site] = None

                    if current_timeslot > update.timeslot_idx:
                        _logger.warning(
                            f'Plan update at {site.name} for night {night_idx} for {update.event.__class__.__name__}'
                            f' scheduled for timeslot {update.timeslot_idx}, but now timeslot is {current_timeslot}.')

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
                        _logger.debug(f'Time accounting: site {site_name} for night {night_idx} {ta_description}')
                        collector.time_accounting(plans=plans,
                                                  sites=frozenset({site}),
                                                  end_timeslot_bounds=end_timeslot_bounds)

                        if update.done:
                            # In the case of the morning twilight, which is the only thing that will
                            # be represented here by update.done, we add no plans (None) since the plans
                            # generated up until the terminal time slot will have been added by the event
                            # that caused them.
                            nightly_timeline.add(NightIndex(night_idx),
                                                 site,
                                                 current_timeslot,
                                                 update.event,
                                                 None)

                    # Get a new selection and request a new plan if the night is not done.
                    if not update.done:
                        _logger.debug(f'Calculating selection for {site_name} for night {night_idx} '
                                      f'starting at time slot {current_timeslot}.')

                        # If the site is blocked, we do not perform a selection or optimizer run for the site.
                        if change_monitor.is_site_unblocked(site):
                            selection = selector.select(night_indices=night_indices,
                                                        sites=frozenset([site]),
                                                        starting_time_slots={site: {night_idx: current_timeslot
                                                                                    for night_idx in night_indices}},
                                                        ranker=ranker)

                            # Right now the optimizer generates List[Plans], a list of plans indexed by
                            # every night in the selection. We only want the first one, which corresponds
                            # to the current night index we are looping over.
                            _logger.debug(f'Running optimizer for {site_name} for night {night_idx} '
                                          f'starting at timeslot {current_timeslot}.')
                            plans = optimizer.schedule(selection)[0]
                            nightly_timeline.add(NightIndex(night_idx),
                                                 site,
                                                 current_timeslot,
                                                 update.event,
                                                 plans[site])

                        else:
                            # The site is blocked.
                            _logger.debug(f'Site {site_name} for {night_idx} blocked at timeslot {current_timeslot}.')
                            nightly_timeline.add(NightIndex(night_idx),
                                                 site,
                                                 current_timeslot,
                                                 update.event,
                                                 None)

                    # Update night_done based on time update record.
                    night_done = update.done

                # We have processed all events for this timeslot and performed an update if necessary.
                # Advance the current time.
                current_timeslot += 1

            # Process any events still remaining, with the intent of unblocking faults and weather closures.
            eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
            while events_by_night.has_more_events():
                event = events_by_night.pop_next_event()
                event.to_timeslot_idx(eve_twi_time, time_slot_length)
                _logger.warning(f'Site {site_name} on night {night_idx} has event after morning twilight: {event}')
                change_monitor.process_event(site, event, None, night_idx)

                # Timeslot will be after final timeslot because this event is scheduled later.
                nightly_timeline.add(NightIndex(night_idx), site, current_timeslot, event, None)

            # The site should no longer be blocked.
            if not change_monitor.is_site_unblocked(site):
                _logger.warning(f'Site {site_name} is still blocked after all events on night {night_idx} processed.')

        # Piece together the plans for the night to get the overall plans.
        # This is rather convoluted because of the confusing relationship between Plan, Plans, and NightlyTimeline.
        night_events = {site: collector.get_night_events(site) for site in collector.sites}
        night_conditions = {site: initial_variants[site][night_idx] for site in collector.sites}
        final_plans = Plans(night_events, night_conditions, NightIndex(night_idx))
        for site in collector.sites:
            calculated_plan = nightly_timeline.get_final_plan(NightIndex(night_idx), site)
            if calculated_plan is not None:
                final_plans[site] = nightly_timeline.get_final_plan(NightIndex(night_idx), site)
        overall_plans[night_idx] = final_plans

    plan_summary = StatCalculator.calculate_timeline_stats(nightly_timeline,
                                                           night_indices,
                                                           sites, collector)

    # Make sure we have a list of the final plans sorted by night index.
    plans_list = [p for _, p in sorted(overall_plans.items())]
    # plan_summary = StatCalculator.calculate_plans_stats(overall_plans, collector)
    nightly_timeline.display()

    print('++++ FINAL PLANS ++++')
    print_plans(plans_list)  # List[Plans]
    print('DONE')
