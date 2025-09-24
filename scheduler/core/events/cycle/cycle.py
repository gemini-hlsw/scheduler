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
            ta_description = f'up to timeslot {current_timeslot}.'

        _logger.info(f'Time accounting: site {site.site_name} for night {night_idx} {ta_description}')

        # Run time accounting
        self.scp.collector.time_accounting(
            plans=plans,
            sites=frozenset({site}),
            end_timeslot_bounds=end_timeslot_bounds
        )

        if update.done:
            # For morning twilight, add final plan showing all observations
            _logger.debug('Night done. Wrapping up final plan')
            # final_plan = nightly_timeline.get_final_plan(NightIndex(night_idx),
            #                                            site,
            #                                           self.change_monitor.is_site_unblocked(site))
            final_plan = self.scp.collector.final_plans[night_idx][site]
            final_plan.conditions = self.scp.selector._variant_snapshot_per_site[site]
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

        _logger.info(
            f'Retrieving selection for {site.site_name} for night {night_idx} '
            f'starting at time slot {current_timeslot}.'
        )

        # If the site is unblocked, perform selection and optimization and add plan to night
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
            # The site is blocked, add a None plan
            _logger.debug(f'Site {site.site_name} for {night_idx} blocked at timeslot {current_timeslot}.')
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
        eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)

        while events_by_night.has_more_events():
            event = events_by_night.pop_next_event()
            event.to_timeslot_idx(eve_twi_time, time_slot_length)
            _logger.warning(f'Site {site.site_name} on night {night_idx} has event after morning twilight: {event}')

            self.change_monitor.process_event(site, event, None, night_idx)

            # Timeslot will be after final timeslot because this event is scheduled later
            nightly_timeline.add(NightIndex(night_idx), site, current_timeslot, event, None)

    def _process_current_events(self,
                                site: Site,
                                night_idx: NightIndex,
                                night_start: datetime,
                                time_slot_length: timedelta,
                                events_by_night: NightEventQueue,
                                plans: Plans,
                                update: TimeCoordinateRecord,
                                previous_event_timeslot: TimeslotIndex) -> Tuple[Event, Optional[TimeslotIndex]]:
        """Process events that occur at the current timeslot.
        Args:
            site (Site): Site being processed
            night_idx (NightIndex): Index of the night
            night_start (datetime): Start time of the night
            time_slot_length (datetime): Length of each timeslot
            events_by_night (NightEventQueue): Queue of events
            plans (Plans): Current observation plans
            update: The time record being processed
            previous_event_timeslot (int): Timeslot of the previous event
        Returns:
            Tuple of (update, current_timeslot) - updated values
        """
        # Get next event
        event = events_by_night.top_event()
        # Get current timeslot
        current_timeslot = event.to_timeslot_idx(night_start, time_slot_length)

        # First event of the night
        if update is None:
            # Process event and remove it from the queue
            events_by_night.pop_next_event()
            update = self.change_monitor.process_event(site, event, plans, night_idx)

        # Subsequent events
        else:
            # Previous event is not processed yet and should be done before the new event
            if previous_event_timeslot < update.timeslot_idx <= current_timeslot:
                current_timeslot = update.timeslot_idx

            # Previous event should be discarded since new event is processed before it
            else:
                # Process event and remove it from the queue
                events_by_night.pop_next_event()
                update = self.change_monitor.process_event(site, event, plans, night_idx)

        return update, current_timeslot

    def _handle_updates(self,
                        site: Site,
                        night_idx: NightIndex,
                        current_timeslot: TimeslotIndex,
                        update: TimeCoordinateRecord,
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
        if update is not None and current_timeslot >= update.timeslot_idx:
            # Determine end timeslot bounds for time accounting
            end_timeslot_bounds = {} if update.done else {site: update.timeslot_idx}

            # If there was an old plan and time accounting is needed, process it
            if plans is not None and update.perform_time_accounting:
                self._perform_time_accounting(
                    site,
                    night_idx,
                    update,
                    end_timeslot_bounds,
                    plans,
                    current_timeslot,
                    nightly_timeline
                )

            # Get a new selection and request a new plan if the night is not done
            if not update.done:
                plans = self._create_new_plan(
                    site,
                    night_idx,
                    current_timeslot,
                    update,
                    plans,
                    nightly_timeline
                )
        
        return plans

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
        time_slot_length = self.scp.collector.time_slot_length.to_datetime()

        # Get night events and key times
        night_events = self.scp.collector.get_night_events(site)
        # We need the start of the night for checking if an event has been reached.
        night_start = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)

        # Get event queue for the night and site
        events_by_night = self.queue.get_night_events(night_idx, site)
        if events_by_night.is_empty():
            raise RuntimeError(f'No events for site {site.site_name} for night {night_idx}.')

        # Initialize tracking variables
        current_timeslot: TimeslotIndex = TimeslotIndex(0)
        plans: Optional[Plans] = None
        update = None
        previous_event_timeslot = TimeslotIndex(0)

        # Set the initial variant for the site for the night. This may have been set above by weather
        # information obtained before or at the start of the night, and if not, then the lookup will give None,
        # which will reset to the default values as defined in the Selector.
        morn_twi_time = events_by_night.events[0].time
        initial_variant = self.scp.collector.sources.origin.env.get_initial_conditions(
            site,
            morn_twi_time.date()
        )
        self.scp.selector.update_site_variant(site, initial_variant)
        _logger.debug(f'Resetting {site.site_name} weather to initial values for night...')

        # Loop through events until there are no more events
        while events_by_night.has_more_events():
            update, current_timeslot = self._process_current_events(
                site,
                night_idx,
                night_start,
                time_slot_length,
                events_by_night,
                plans,
                update,
                previous_event_timeslot
            )

            plans = self._handle_updates(
                site,
                night_idx,
                current_timeslot,
                update,
                plans,
                nightly_timeline
            )

            # Set the previous event timeslot
            previous_event_timeslot = current_timeslot
        
        # Process after twilight remaining events (if any)
        self._process_remaining_events(
            site,
            night_idx,
            night_events,
            events_by_night,
            time_slot_length,
            nightly_timeline,
            current_timeslot
        )

        # The site should no longer be blocked.
        if not self.change_monitor.is_site_unblocked(site):
            _logger.warning(f'Site {site.site_name} is still blocked after all events on night {night_idx} processed.')
