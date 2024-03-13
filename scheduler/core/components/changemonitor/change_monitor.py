# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bisect
from dataclasses import dataclass
from typing import final, Optional

from astropy.coordinates import Angle
import numpy as np
from lucupy.minimodel import NightIndex, ObservationClass, Site, TimeslotIndex, Variant

from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.eventsqueue import Event, EveningTwilightEvent, MorningTwilightEvent, WeatherChangeEvent
from scheduler.core.plans import Plans
from scheduler.services.logger_factory import create_logger
from .time_coordinate_record import TimeCoordinateRecord


__all__ = [
    'ChangeMonitor',
]

_logger = create_logger(__name__)


@final
@dataclass
class ChangeMonitor(SchedulerComponent):
    collector: Collector
    selector: Selector

    def process_event(self,
                      site: Site,
                      event: Event,
                      plans: Optional[Plans],
                      night_idx: NightIndex) -> Optional[TimeCoordinateRecord]:
        """
        TODO: Might want to make return type a Tuple[NightIndex, TimeslotIndex].

        Given an event occurring at a given site and an optional plan running on the site, determine
        the next timeslot where the plan should be recalculated, if any.

        The plan is optional, because for EveningTwilightEvent, no plan has yet been computed and thus None should
        be passed in.

        If the ChangeMonitor determines that a new plan should be calculated, then a timeslot index indicating
        when the plan should be calculated is returned, and if no new plan should be calculated, then None is returned.

        :param site: the site at which the event occurred
        :param event: the event that occurred
        :param plans: the plans that are currently in action (if any) for the night, which consist of a plan per site
        :param night_idx: the night index

        :return: a time coordinate record which provides information about when the next plan should be computed
                 (if any), if a night is done, and if time accounting should be performed
        """
        # Convert the site to a name here in case we want to use site.name (key name) or site.site_name (long form).
        site_name = site.name

        # Translate the local event time to Scheduler time coordinates.
        night_events = self.collector.get_night_events(site)
        twi_eve = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
        event_timeslot = event.to_timeslot_idx(twi_eve, self.collector.time_slot_length.to_datetime())

        # Check that the event is valid.
        # if event_night > curr_night:
        #     raise ValueError(f'Site {site_name} is running night index {curr_night}, but received event {event} '
        #                      f'for night index {event_night}.')
        # if (event_night < curr_night or
        #         (event_night == curr_night and event_timeslot < curr_timeslot)):
        #     raise ValueError(f'Site {site_name} is running night index {curr_night} and time slot index '
        #                      f'{curr_timeslot}, received event {event} for earlier time: '
        #                      f'night index {event_night} and time slot index {event_timeslot}.')

        num_timeslots_for_night = night_events.num_timeslots_per_night[night_idx]
        last_timeslot_for_night = TimeslotIndex(num_timeslots_for_night - 1)

        # Process the event based on its type:
        match event:
            case EveningTwilightEvent():
                # We always create a plan for the evening twilight event.
                if event_timeslot != 0:
                    _logger.warning(f'EveningTwilightEvent for site {site_name} should be scheduled for timeslot 0, '
                                    f'but is scheduled for timeslot {event_timeslot}.')
                # We do not perform any time accounting for the evening twilight.
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=event_timeslot,
                                            perform_time_accounting=False)

            case MorningTwilightEvent():
                if event_timeslot != last_timeslot_for_night:
                    _logger.warning(f'MorningTwilightEvent for site {site_name} should be scheduled for '
                                    f'timeslot {last_timeslot_for_night} but is scheduled for '
                                    f'timeslot {event_timeslot}.')
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=last_timeslot_for_night,
                                            done=True)

            case WeatherChangeEvent(variant_change=variant_change):
                # Regardless, we want to change the weather values for CC and IQ.
                self.selector.update_site_variant(site, variant_change)

                if plans is None:
                    raise ValueError(f'No plans have been created for night {night_idx}.')

                # Check if there is a visit running now.
                plan = plans[site]
                if plan is None:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Sort the visits by start time and find the one (if any) that happens just before this event.
                sorted_visits = sorted(plan.visits, key=lambda v: v.start_time_slot)
                visit_idx = bisect.bisect_right([v.start_time_slot for v in sorted_visits], event_timeslot) - 1
                visit = None if visit_idx < 0 else sorted_visits[visit_idx]

                # If there are no visits in progress, then recalculate now.
                if visit is None:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Check if event occurs after the calculated visit is already over. If so, recalculate now.
                visit_end_time_slot = visit.start_time_slot + visit.time_slots - 1
                if visit_end_time_slot < event_timeslot:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise, we are in the middle of a visit and interrupt it.
                remaining_time_slots = visit_end_time_slot - event_timeslot + 1

                # TODO: This should be more complicated to allow for splitting and to meet requirements.
                # TODO: Talk to Bryan about how to go about this.
                obs = Collector.get_observation(visit.obs_id)

                # Most restrictive conditions.
                mrc = obs.constraints.conditions

                # TODO: This code is somewhat duplicated from Selector. See if we can simplify it, although in this
                # TODO: case, it is for a single night instead of all nights.
                target_info = Collector.get_target_info(obs.id)
                if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
                    neg_ha = target_info[night_idx].hourangle[0].value < 0
                else:
                    neg_ha = False
                too_type = obs.too_type

                # Get the actual conditions for the time slots remaining for the observation.
                # We compare from the current time slot to the end time slot.

                # TODO: We are getting 1-off errors sometimes by doing the weather lookups this way in the size of
                # TODO: remaining_time_slots.
                # start_time = night_events.times[event_night][event_timeslot]
                # end_time = night_events.times[event_night][end_time_slot]
                #
                # # Get the actual variant from the Weather forecast service.
                # actual_conditions = self.collector.sources.origin.env.get_actual_conditions_variant(obs.site,
                #                                                                                     start_time,
                #                                                                                     end_time)
                #
                # # TODO: Hack to make test cases pass.
                # if remaining_time_slots != len(actual_conditions.cc):
                #     _logger.error(f'Expected {remaining_time_slots} entries in CC, got {len(actual_conditions.cc)}.')
                # if remaining_time_slots != len(actual_conditions.iq):
                #     _logger.error(f'Expected {remaining_time_slots} entries in IQ, got {len(actual_conditions.iq)}.')
                # if remaining_time_slots != len(actual_conditions.wind_dir):
                #     _logger.error(f'Expected {remaining_time_slots} entries in wind direction, got '
                #                   f'{len(actual_conditions.wind_dir)}.')
                # if remaining_time_slots != len(actual_conditions.wind_spd):
                #     _logger.error(f'Expected {remaining_time_slots} entries in wind speed, got '
                #                   f'{len(actual_conditions.wind_spd)}.')
                # remaining_time_slots = max(remaining_time_slots, len(actual_conditions.cc))

                # Create a variant representing the weather values.
                cc = np.array([variant_change.cc] * remaining_time_slots)
                iq = np.array([variant_change.iq] * remaining_time_slots)
                wind_dir = Angle(np.full(remaining_time_slots, variant_change.wind_dir.value),
                                 unit=variant_change.wind_dir.unit)
                wind_spd = np.full(remaining_time_slots, variant_change.wind_spd.value) * variant_change.wind_spd.unit
                actual_conditions = Variant(cc=cc,
                                            iq=iq,
                                            wind_dir=wind_dir,
                                            wind_spd=wind_spd)

                # Compare the conditions with those required by the observation.
                # They will all be 0 or 1 since the Variant has consistent values, but check regardless if there
                # is a timeslot where we cannot execute the visit. If so, recalculate now.
                slot_values = Selector.match_conditions(mrc, actual_conditions, neg_ha, too_type)
                if not np.all(slot_values > 0):
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise, we can finish the observation. Start the weather change at time slot after the visit ends.
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=TimeslotIndex(visit_end_time_slot + 1))

            # For now, for all other events, just recalculate immediately.
            case _:
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=event_timeslot)
