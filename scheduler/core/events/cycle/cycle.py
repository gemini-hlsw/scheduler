

class EventCycle:

    def __init__(self, params):
        self.params = params

    def run(self, scp):

        next_update = {site: None for site in self.params.sites}
        # We need the start of the night for checking if an event has been reached.
        # Next update indicates when we will recalculate the plan.
        night_events = scp.collector.get_night_events(site)
        night_start = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
        next_update[site] = None

        while not night_done:
            # If our next update isn't done, and we are out of events, we're missing the morning twilight.
            if next_event is None and events_by_night.is_empty():
                raise RuntimeError(f'No morning twilight found for site {site_name} for night {night_idx}.')

            if next_event_timeslot is None or current_timeslot >= next_event_timeslot:

                if not events_by_night.has_more_events():
                    # Check if there are no more events so it won't enter the loop behind
                    break
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
                        # Things happening after the EveTwilight fall here as the current_timeslot start at 0.
                        # We could handle stuff
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
                    time_record = self.change_monitor.process_event(site, top_event, plans, night_idx)
                    if time_record is not None:
                        # In the case that:
                        # * there is no next update scheduled; or
                        # * this update happens before the next update
                        # then set to this update.
                        if next_update[site] is None or time_record.timeslot_idx <= next_update[site].timeslot_idx:
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
                    scp.collector.time_accounting(plans=plans,
                                                  sites=frozenset({site}),
                                                  end_timeslot_bounds=end_timeslot_bounds)

                    if update.done:
                        # In the case of the morning twilight, which is the only thing that will
                        # be represented here by update.done, we add the final plan that shows all the
                        # observations that were actually visited in that night.
                        final_plan = nightly_timeline.get_final_plan(NightIndex(night_idx), site)
                        nightly_timeline.add(NightIndex(night_idx),
                                             site,
                                             current_timeslot,
                                             update.event,
                                             final_plan)

                # Get a new selection and request a new plan if the night is not done.
                if not update.done:
                    _logger.debug(f'Retrieving selection for {site_name} for night {night_idx} '
                                  f'starting at time slot {current_timeslot}.')

                    # If the site is blocked, we do not perform a selection or optimizer run for the site.
                    if self.change_monitor.is_site_unblocked(site):
                        plans = scp.run(site, night_indices, current_timeslot, ranker)
                        nightly_timeline.add(NightIndex(night_idx),
                                             site,
                                             current_timeslot,
                                             update.event,
                                             plans[site])
                    else:
                        # The site is blocked.
                        _logger.debug(
                            f'Site {site_name} for {night_idx} blocked at timeslot {current_timeslot}.')
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
            self.change_monitor.process_event(site, event, None, night_idx)

            # Timeslot will be after final timeslot because this event is scheduled later.
            nightly_timeline.add(NightIndex(night_idx), site, current_timeslot, event, None)

        # The site should no longer be blocked.
        if not self.change_monitor.is_site_unblocked(site):
            _logger.warning(f'Site {site_name} is still blocked after all events on night {night_idx} processed.')
