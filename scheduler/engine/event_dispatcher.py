from scheduler.core.events.queue import WeatherChangeEvent


class EventDispatcher:

    def __init__(self):
        self._handlers = {
            WeatherChangeEvent: self.weather_change_event
        }


    def weather_change_event(self, event):
        """
        Compute a new plan based on the given event.

        Args:
            event (Event): The event to compute the plan for.
        Returns:
            NewPlansRT: The new plan for the event.
        """
        # Get the timeslots associated with the sites with format
        # {site: {0: current_timeslot}}

        await self.build()
        self.init_variant()

        start_timeslot = {}
        for site in self.params.sites:
            night_start_time = self.scp.collector.night_events[site].times[0][0]
            utc_night_start = night_start_time.utc.to_datetime(timezone=datetime.timezone.utc)

            event_timeslot = to_timeslot_idx(
                # event.time, all event need to happen in the test range for now
                utc_night_start + datetime.timedelta(hours=3),
                utc_night_start,
                self.scp.collector.time_slot_length.to_datetime()
            )
            start_timeslot[site] = {np.int64(0): event_timeslot}

        plans = self.scp.run_rt(start_timeslot)

        for site in self.params.sites:
            plans.plans[site].night_stats = NightStats({}, 0.0, 0, {}, {})
            plans.plans[site].alt_degs = []
            # Calculate altitude data
            for visit in plans.plans[site].visits:
                ti = self.scp.collector.get_target_info(visit.obs_id)
                end_time_slot = visit.start_time_slot + visit.time_slots
                values = ti[plans.night_idx].alt[visit.start_time_slot: end_time_slot]
                alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
                plans.plans[site].alt_degs.append(alt_degs)
        splans = SPlans.from_computed_plans(plans, self.params.sites)

        return NewPlansRT(night_plans=splans)