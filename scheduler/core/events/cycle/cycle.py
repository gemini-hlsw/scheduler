from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import numpy as np
from lucupy.minimodel import TimeslotIndex, NightIndex, Site

from scheduler.core.calculations import NightEvents
from scheduler.core.components.changemonitor import ChangeMonitor, TimeCoordinateRecord
from scheduler.core.events.queue import Event, NightlyTimeline, EventQueue, NightEventQueue
from scheduler.core.plans import Plans
from scheduler.core.scp import SCP
from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)

NextUpdate = Dict[Site, Optional[TimeCoordinateRecord]]

__all__ = ["EventCycle"]

class EventCycle:
    """
    Attributes:
     params (SchedulerParams): Scheduler parameters
     queue (EventQueue): Event queue that orders the events chronologically
     scp (SCP): The Scheduler Core Pipeline that allows creating plans.

    """
    def __init__(self,
                 params,
                 queue: EventQueue,
                 scp: SCP):
        self.params = params
        self.queue = queue
        self.scp = scp
        self.change_monitor = ChangeMonitor(scp.collector, scp.selector)

    def _process_current_events(self,
                                site: Site,
                                night_idx: NightIndex,
                                night_start: datetime,
                                time_slot_length: timedelta,
                                events_by_night: NightEventQueue,
                                plans: Plans,
                                current_timeslot: TimeslotIndex,
                                next_update: NextUpdate,
                                next_event: Event,
                                next_event_timeslot: TimeslotIndex) -> Tuple[Event, Optional[TimeslotIndex]]:
        """Process events that occur at the current timeslot.

        Args:
            site (Site): Site being processed
            night_idx (NightIndex): Index of the night
            night_start (datetime): Start time of the night
            time_slot_length (datetime): Length of each timeslot
            events_by_night (NightEventQueue): Queue of events
            plans (Plans): Current observation plans
            current_timeslot (int): Current timeslot being processed
            next_update: Dict mapping sites to their next update time
            next_event (Event): Next event to be processed
            next_event_timeslot (TimeslotIndex): Timeslot of the next event

        Returns:
            Tuple of (next_event, next_event_timeslot) - updated values
        """

        site_name = site.site_name

        # If our next update isn't done, and we are out of events, we're missing the morning twilight
        if next_event is None and events_by_night.is_empty():
            raise RuntimeError(f'No morning twilight found for site {site_name} for night {night_idx}.')

        if next_event_timeslot is None or current_timeslot >= next_event_timeslot:
            # Stop if there are no more events
            if not events_by_night.has_more_events():
                # This should break the upper while
                return next_event, next_event_timeslot

            # Process events for this timeslot
            while events_by_night.has_more_events():
                top_event = events_by_night.top_event()
                top_event_timeslot = top_event.to_timeslot_idx(night_start, time_slot_length)
                _logger.debug(f'Top event: {top_event.description} at {top_event_timeslot}')

                # If we don't know the next event timeslot, set it
                if next_event_timeslot is None:
                    next_event_timeslot = top_event_timeslot
                    next_event = top_event

                if current_timeslot > top_event_timeslot:
                    _logger.warning(
                        f'Received event for {site_name} for night idx {night_idx} at timeslot '
                        f'{top_event_timeslot} < current time slot {current_timeslot}.')

                # The next event happens in the future, so record that time and break the loop
                if top_event_timeslot > current_timeslot:
                    next_event_timeslot = top_event_timeslot
                    break

                # Process event that occurs at this time slot
                events_by_night.pop_next_event()
                _logger.info(
                    f'Received event for site {site_name} for night idx {night_idx} to be processed '
                    f'at timeslot {top_event_timeslot}: {top_event.__class__.__name__}'
                )

                # Process the event to determine when to recalculate the plan
                time_record = self.change_monitor.process_event(site, top_event, plans, night_idx)
                if time_record is not None:
                    # Update next_update if this event requires an earlier update
                    if (next_update[site] is None or
                            time_record.timeslot_idx <= next_update[site].timeslot_idx):
                        next_update[site] = time_record
                        _logger.debug(
                            f'Next update for site {site_name} scheduled at '
                            f'timeslot {next_update[site].timeslot_idx}'
                        )
        return next_event, next_event_timeslot

    def _handle_updates(self,
                        site: Site,
                        night_idx: NightIndex,
                        current_timeslot: TimeslotIndex,
                        next_update: NextUpdate,
                        plans: Plans,
                        nightly_timeline: NightlyTimeline):
        """Handle any scheduled plan updates for the current timeslot.

        Args:
            site (site): Site being processed
            night_idx (NightIndex): Index of the night
            current_timeslot (TimeslotIndex): Current timeslot being processed
            next_update (NextUpdate): Dict mapping sites to their next update time
            plans (Plans): Current observation plans
            nightly_timeline: Timeline to add events to

        Returns:
           Tuple of (bool, plans): True if night is done, False otherwise and updated plans.
        """
        night_done = False

        # If there is a next update, and we have reached its time, then perform it
        if next_update[site] is not None and current_timeslot >= next_update[site].timeslot_idx:
            # Remove the update and perform it
            update = next_update[site]
            next_update[site] = None

            if current_timeslot > update.timeslot_idx:
                _logger.warning(
                    f'Plan update at {site.name} for night {night_idx} for {update.event.__class__.__name__}'
                    f' scheduled for timeslot {update.timeslot_idx}, but now timeslot is {current_timeslot}.'
                )

            # Determine end timeslot bounds for time accounting
            end_timeslot_bounds = {} if update.done else {site: update.timeslot_idx}

            # If there was an old plan and time accounting is needed, process it
            if plans is not None and update.perform_time_accounting:
                self._perform_time_accounting(site,
                                              night_idx,
                                              update,
                                              end_timeslot_bounds,
                                              plans,
                                              current_timeslot,
                                              nightly_timeline)

            # Get a new selection and request a new plan if the night is not done
            if not update.done:
                plans = self._create_new_plan(site,
                                      night_idx,
                                      current_timeslot,
                                      update,
                                      plans,
                                      nightly_timeline)

            # Update night_done based on time update record
            night_done = update.done

        return night_done, plans

    def _perform_time_accounting(self,
                                 site: Site,
                                 night_idx: NightIndex,
                                 update: TimeCoordinateRecord,
                                 end_timeslot_bounds: Dict[Site, TimeslotIndex],
                                 plans: Plans,
                                 current_timeslot: TimeslotIndex,
                                 nightly_timeline: NightlyTimeline):

        """Perform time accounting for executed plans.

        Also add the final plan for the Morning twilight.

        Args:
            site: Site being processed
            night_idx: Index of the night
            update: The update record
            end_timeslot_bounds: End bounds for time accounting
            plans: Current observation plans
            current_timeslot: Current timeslot
            nightly_timeline: Timeline to add events to
        """
        if update.done:
            ta_description = 'for rest of night.'
        else:
            ta_description = f'up to timeslot {update.timeslot_idx}.'

        _logger.info(f'Time accounting: site {site.site_name} for night {night_idx} {ta_description}')

        self.scp.collector.time_accounting(
            plans=plans,
            sites=frozenset({site}),
            end_timeslot_bounds=end_timeslot_bounds
        )

        if update.done:
            # For morning twilight, add final plan showing all observations
            _logger.debug('Night done. Wrapping up final plan')
            final_plan = nightly_timeline.get_final_plan(NightIndex(night_idx),
                                                         site,
                                                         self.change_monitor.is_site_unblocked(site))
            nightly_timeline.add(
                NightIndex(night_idx),
                site,
                current_timeslot,
                update.event,
                final_plan
            )

    def _create_new_plan(self,
                         site: Site,
                         night_idx: NightIndex,
                         current_timeslot: TimeslotIndex,
                         update: TimeCoordinateRecord,
                         plans: Plans,
                         nightly_timeline: NightlyTimeline):
        """Create a new observation plan.

        Args:
            site: Site being processed
            night_idx: Night being processed
            current_timeslot: Current timeslot
            update: The time record being processed
            nightly_timeline: Timeline to add events to

        Returns:
            New plans or None if site is blocked
        """
        night_indices = np.array([night_idx])
        site_name = site.site_name
        # plans = None

        _logger.info(
            f'Retrieving selection for {site_name} for night {night_idx} '
            f'starting at time slot {current_timeslot}.'
        )

        # If the site is unblocked, perform selection and optimization
        if self.change_monitor.is_site_unblocked(site):
            plans = self.scp.run(site, night_indices, current_timeslot)
            nightly_timeline.add(
                NightIndex(night_idx),
                site,
                current_timeslot,
                update.event,
                plans[site]
            )
        else:
            # The site is blocked
            _logger.debug(f'Site {site_name} for {night_idx} blocked at timeslot {current_timeslot}.')
            nightly_timeline.add(
                NightIndex(night_idx),
                site,
                current_timeslot,
                update.event,
                None
            )
        return plans

    def _process_remaining_events(self,
                                  site: Site,
                                  night_idx: NightIndex,
                                  night_events: NightEvents,
                                  events_by_night: NightEventQueue,
                                  time_slot_length: timedelta,
                                  nightly_timeline: NightlyTimeline,
                                  current_timeslot: TimeslotIndex):
        """Process any events still remaining after night is done.

        Args:
            site (Site): Site being processed
            night_idx (NightIndex): Index of the night
            night_events (NightEvents): Night Events for the night
            events_by_night (NightEventQueue): Queue of events
            time_slot_length (timedelta): Length of each timeslot
            nightly_timeline (NightlyTimeline): Timeline to add events to
            current_timeslot (int): Current timeslot
        """
        site_name = site.site_name

        eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)

        while events_by_night.has_more_events():
            event = events_by_night.pop_next_event()
            event.to_timeslot_idx(eve_twi_time, time_slot_length)
            _logger.warning(f'Site {site_name} on night {night_idx} has event after morning twilight: {event}')

            self.change_monitor.process_event(site, event, None, night_idx)

            # Timeslot will be after final timeslot because this event is scheduled later
            nightly_timeline.add(NightIndex(night_idx), site, current_timeslot, event, None)

    def run(self, site: Site, night_idx: NightIndex, nightly_timeline: NightlyTimeline):
        """Executes the Event cycle for a specific site and night.

        Args:
            site (Site): Site to process events from the night.
            night_idx (NightIndex): Night to process events from.
            nightly_timeline (NightlyTimeline): Records all the events and their corresponding plans.

        Raises:
            RuntimeError: If required events are missing
        """

        # Initialize key components
        site_name = site.site_name
        time_slot_length = self.scp.collector.time_slot_length.to_datetime()

        # Get night events and key times
        # We need the start of the night for checking if an event has been reached.
        # Next update indicates when we will recalculate the plan.
        night_events = self.scp.collector.get_night_events(site)
        # night_start is on local date?
        night_start = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)  # this on local date
        events_by_night = self.queue.get_night_events(night_idx, site)
        if events_by_night.is_empty():
            raise RuntimeError(f'No events for site {site_name} for night {night_idx}.')

        # Initialize tracking variables
        night_done = False
        next_event: Optional[Event] = None
        next_event_timeslot: Optional[TimeslotIndex] = None
        next_update = {site: None for site in self.params.sites}
        current_timeslot: TimeslotIndex = TimeslotIndex(0)
        plans: Optional[Plans] = None

        # Set the initial variant for the site for the night. This may have been set above by weather
        # information obtained before or at the start of the night, and if not, then the lookup will give None,
        # which will reset to the default values as defined in the Selector.
        morn_twi_time = events_by_night.events[0].time
        initial_variant = self.scp.collector.sources.origin.env.get_initial_conditions(site, morn_twi_time.date())
        self.scp.selector.update_site_variant(site, initial_variant)
        _logger.debug(f'Resetting {site.name} weather to initial values for night...')

        while not night_done:
            # Process current event at the current time
            next_event, next_event_timeslot = self._process_current_events(site,
                                           night_idx,
                                           night_start,
                                           time_slot_length,
                                           events_by_night,
                                           plans,
                                           current_timeslot,
                                           next_update,
                                           next_event,
                                           next_event_timeslot)
            # Check if it's time to update the plan
            night_done, plans = self._handle_updates(site,
                                             night_idx,
                                             current_timeslot,
                                             next_update,
                                             plans,
                                             nightly_timeline)

            # We have processed all events for this timeslot and performed an update if necessary.
            # Advance the current time.
            current_timeslot += 1

            if next_event_timeslot < current_timeslot:
                _logger.error(f'Next event timeslot {next_event_timeslot} is before current timeslot {current_timeslot} for site {site_name} on night {night_idx}.')
                break

        self._process_remaining_events(site,
                                       night_idx,
                                       night_events,
                                       events_by_night,
                                       time_slot_length,
                                       nightly_timeline,
                                       current_timeslot)

        # The site should no longer be blocked.
        if not self.change_monitor.is_site_unblocked(site):
            _logger.warning(f'Site {site_name} is still blocked after all events on night {night_idx} processed.')
