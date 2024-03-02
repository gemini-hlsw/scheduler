# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bisect
from dataclasses import dataclass
from typing import final, Optional

import numpy as np
from lucupy.minimodel import Site, TimeslotIndex, ObservationClass, Variant

from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.events_queue import Event, EveningTwilightEvent, MorningTwilightEvent, WeatherChangeEvent
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
                      plans: Optional[Plans]) -> Optional[TimeCoordinateRecord]:
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

        :return: a time coordinate record which provides information about when the next plan should be computed
                 (if any), if a night is done, and if time accounting should be performed
        """
        # Convert the site to a name here in case we want to use site.name (key name) or site.site_name (long form).
        site_name = site.name

        # Translate the local event time to Scheduler time coordinates.
        night_events = self.collector.get_night_events(site)
        event_night, event_timeslot = night_events.local_dt_to_time_coords(event.time)

        # Check that the event is valid.
        # if event_night > curr_night:
        #     raise ValueError(f'Site {site_name} is running night index {curr_night}, but received event {event} '
        #                      f'for night index {event_night}.')
        # if (event_night < curr_night or
        #         (event_night == curr_night and event_timeslot < curr_timeslot)):
        #     raise ValueError(f'Site {site_name} is running night index {curr_night} and time slot index '
        #                      f'{curr_timeslot}, received event {event} for earlier time: '
        #                      f'night index {event_night} and time slot index {event_timeslot}.')

        num_timeslots_for_night = night_events.num_timeslots_per_night[event_night]
        last_timeslot_for_night = TimeslotIndex(num_timeslots_for_night - 1)

        # Process the event based on its type:
        match event:
            case EveningTwilightEvent():
                # We always create a plan for the evening twilight event.
                if event_timeslot != 0:
                    _logger.warning(f'EveningTwilightEvent for site {site_name} should be scheduled for timeslot 0, '
                                    f'but is scheduled for timeslot {event_timeslot}.')
                    print(f'EveningTwilightEvent for site {site_name} should be scheduled for timeslot 0, '
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
                    print(f'MorningTwilightEvent for site {site_name} should be scheduled for '
                          f'timeslot {last_timeslot_for_night} but is scheduled for '
                          f'timeslot {event_timeslot}.')
                # TODO: Do we want to return any other information?
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=last_timeslot_for_night,
                                            done=True)

            case WeatherChangeEvent(variant_change=variant_change):
                # Regardless, we want to change the weather values for CC and IQ.
                self.selector.update_variant(site, variant_change)

                # There should be plans already established.
                assert plans is not None, f"WeatherChangeEvent received for {event_night}, but no plans have been made."

                # Check if there is a visit running now.
                plan = plans[site]
                sorted_visits = sorted(plan.visits, key=lambda v: v.start_time_slot)
                visit_idx = bisect.bisect_right([v.start_time_slot for v in sorted_visits], event_timeslot)
                visit = None if visit_idx < 0 else sorted_visits[visit_idx]

                # There are no visits currently in progress, so immediately calculate new plan and do TA.
                if visit_idx is None or visit.start_time_slot + visit.time_slots < event_timeslot:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise, we are in the middle of a visit.
                end_time_slot = visit.start_time_slot + visit.time_slots
                remaining_time_slots = end_time_slot - event_timeslot + 1

                # TODO: This should be more complicated to allow for splitting and to meet requirements.
                # TODO: Talk to Bryan about how to go about this.
                obs = Collector.get_observation(visit.obs_id)

                # Most restrictive conditions.
                mrc = obs.constraints.conditions

                # TODO: This code is somewhat duplicated from Selector. See if we can simplify it, although in this
                # TODO: case, it is for a single night instead of all nights.
                target_info = Collector.get_target_info(obs.id)
                if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
                    neg_ha = target_info[event_night].hourangle[0].value < 0
                else:
                    neg_ha = False
                too_type = obs.too_type

                # TODO: We have to rethink how we handle the weather.
                # TODO: In OCS, the forecast will be perfect for the entire night since it is based on historical data.
                # TODO: ASK SCIENCE: do we want weather changes to be predicted (forecast - taken from known data) or
                # TODO: "surprises" to the Scheduler? In the former case, we will never get a WeatherChangeEvent
                # TODO: unless we deliberately introduce one. In the latter case, we will only use WeatherChangeEvents
                # TODO: to constantly surprise the Scheduler.

                # TODO: In GPP, we will retrieve forecast at beginning of night.
                # TODO: If we get a subscription for weather change or if observer marks weather change, then we will
                # TODO: get a WeatherChangeEvent.

                # TODO: Either way, when a WeatherChangeEvent happens, we should change the Variant for the rest of the
                # TODO: night to have the new VariantChange values.

                # TODO: Right now we are currently in a state of flux between the two:
                # TODO: The Selector keeps a CC and IQ value, but takes a wind direction and wind speed value from
                # TODO: the forecast. The CC and IQ values are consistent and changed only by WeatherChangeEvents,
                # TODO: (although this will presumably change when we revert to the weather service and we take out the
                # TODO: code in the Selector that overwrites the Variant cc and iq arrays), and wind direction and
                # TODO: wind speed are determined from the forecast and do not change.
                # TODO: We must discuss with Science which direction we ultimately want to go.

                # Get the actual conditions for the time slots remaining for the observation.
                # We compare from the current time slot to the end time slot.

                # TODO: We are getting 1-off errors sometimes by doing the weather lookups this way in the size of
                # TODO: remaining_time_slots.
                start_time = night_events.times[event_night][event_timeslot]
                end_time = night_events.times[event_night][end_time_slot]

                # Get the actual variant from the Weather forecast service.
                actual_conditions = self.collector.sources.origin.env.get_actual_conditions_variant(obs.site,
                                                                                                    start_time,
                                                                                                    end_time)

                # TODO: Hack to make test cases pass.
                if remaining_time_slots != len(actual_conditions.cc):
                    _logger.error(f'Expected {remaining_time_slots} entries in CC, got {len(actual_conditions.cc)}.')
                if remaining_time_slots != len(actual_conditions.iq):
                    _logger.error(f'Expected {remaining_time_slots} entries in IQ, got {len(actual_conditions.iq)}.')
                if remaining_time_slots != len(actual_conditions.wind_dir):
                    _logger.error(f'Expected {remaining_time_slots} entries in wind direction, got '
                                  f'{len(actual_conditions.wind_dir)}.')
                if remaining_time_slots != len(actual_conditions.wind_spd):
                    _logger.error(f'Expected {remaining_time_slots} entries in wind speed, got '
                                  f'{len(actual_conditions.wind_spd)}.')
                remaining_time_slots = max(remaining_time_slots, len(actual_conditions.cc))

                # Since a Variant is a frozen dataclass, swap the new values in.
                # Check to make sure the number of values agree.
                assert (len(actual_conditions.cc) == remaining_time_slots,
                        f'Actual conditions have {len(actual_conditions.cc)} timeslots, '
                        f'which does not match {remaining_time_slots} calculated remaining timeslots.')

                # TODO: This is kind of pointless since wind_dir and wind_spd won't remain changed and will always
                # TODO: revert to the forecast when calculating score.
                if variant_change.wind_dir is None:
                    wind_dir = actual_conditions.wind_dir
                else:
                    wind_dir = np.array([variant_change.wind_dir] * remaining_time_slots)
                if variant_change.wind_spd is None:
                    wind_spd = actual_conditions.wind_spd
                else:
                    wind_spd = np.array([variant_change.wind_spd] * remaining_time_slots)
                actual_conditions = Variant(cc=np.array([variant_change.cc] * remaining_time_slots),
                                            iq=np.array([variant_change.iq] * remaining_time_slots),
                                            wind_spd=wind_spd,
                                            wind_dir=wind_dir)

                # Compare the conditions with those required by the observation. If any of them are zero, we can't
                # continue the observation and should just terminate it now.
                slot_values = Selector.match_conditions(mrc, actual_conditions, neg_ha, too_type)

                if not np.all(slot_values > 0):
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise, we can finish the observation. Start the weather change at the end time slot.
                # TODO: end time slot + 1?
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=TimeslotIndex(end_time_slot))

            # For now, for all other events, just recalculate immediately.
            case _:
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=event_timeslot)
