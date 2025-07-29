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
        night_events = self.scp.collector.get_night_events(site)
        # We need the start of the night for checking if an event has been reached.
        night_start = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)

        # Get event queue for the night and site
        events_by_night = self.queue.get_night_events(night_idx, site)
        if events_by_night.is_empty():
            raise RuntimeError(f'No events for site {site_name} for night {night_idx}.')

        # Initialize tracking variables
        current_timeslot: TimeslotIndex = TimeslotIndex(0)
        plans: Optional[Plans] = None
        update = None
        previous_event_timeslot = None

        # Set the initial variant for the site for the night. This may have been set above by weather
        # information obtained before or at the start of the night, and if not, then the lookup will give None,
        # which will reset to the default values as defined in the Selector.
        morn_twi_time = events_by_night.events[0].time
        initial_variant = self.scp.collector.sources.origin.env.get_initial_conditions(site, morn_twi_time.date())
        self.scp.selector.update_site_variant(site, initial_variant)
        _logger.debug(f'Resetting {site.name} weather to initial values for night...')

        # Loop through events until there are no more events
        while events_by_night.has_more_events():
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
                if update.timeslot_idx > previous_event_timeslot and update.timeslot_idx < current_timeslot:
                    current_timeslot = update.timeslot_idx

                # Previous event should be discarded since new event is processed before it
                else:
                    # Process event and remove it from the queue
                    events_by_night.pop_next_event()
                    update = self.change_monitor.process_event(site, event, plans, night_idx)

            # Run plan creation and time accounting if event time is reached
            if update is not None and current_timeslot >= update.timeslot_idx:
                # Determine end timeslot bounds for time accounting
                end_timeslot_bounds = {} if update.done else {site: update.timeslot_idx}

                # If there was an old plan and time accounting is needed, process it
                if plans is not None and update.perform_time_accounting:
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

                # Get a new selection and request a new plan if the night is not done
                if not update.done:
                    night_indices = np.array([night_idx])

                    _logger.info(
                        f'Retrieving selection for {site_name} for night {night_idx} '
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
                        _logger.debug(f'Site {site_name} for {night_idx} blocked at timeslot {current_timeslot}.')
                        nightly_timeline.add(
                            NightIndex(night_idx),
                            site,
                            current_timeslot,
                            update.event,
                            None
                        )

            # Set the previous event timeslot
            previous_event_timeslot = current_timeslot
        
        # Process after twilight remaining events (if any)
        eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)

        while events_by_night.has_more_events():
            event = events_by_night.pop_next_event()
            event.to_timeslot_idx(eve_twi_time, time_slot_length)
            _logger.warning(f'Site {site_name} on night {night_idx} has event after morning twilight: {event}')

            self.change_monitor.process_event(site, event, None, night_idx)

            # Timeslot will be after final timeslot because this event is scheduled later
            nightly_timeline.add(NightIndex(night_idx), site, current_timeslot, event, None)

        # The site should no longer be blocked.
        if not self.change_monitor.is_site_unblocked(site):
            _logger.warning(f'Site {site_name} is still blocked after all events on night {night_idx} processed.')
