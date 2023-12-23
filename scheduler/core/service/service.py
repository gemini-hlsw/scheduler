# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime
from typing import FrozenSet, Optional, Dict

import numpy as np
from astropy.time import Time
from lucupy.minimodel import Site, Semester, NightIndex, TimeslotIndex, CloudCover, ImageQuality

from scheduler.core.builder import Blueprints
from scheduler.core.builder.modes import dispatch_with, SchedulerModes
from scheduler.core.components.collector import Collector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.components.selector import Selector
from scheduler.core.eventsqueue import EventQueue, EveningTwilight, MorningTwilight, WeatherChange
from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
from scheduler.core.sources import Sources
from scheduler.core.statscalculator import StatCalculator


class Service:

    def __init__(self):
        pass

    @staticmethod
    def _setup(night_indices, sites, mode):

        queue = EventQueue(night_indices, sites)
        sources = Sources()
        builder = dispatch_with(mode, sources, queue)
        return builder

    @staticmethod
    def _schedule_nights(night_indices: FrozenSet[NightIndex],
                         sites: FrozenSet[Site],
                         collector: Collector,
                         selector: Selector,
                         optimizer: Optimizer,
                         queue: EventQueue,
                         cc_per_site: Optional[Dict[Site, CloudCover]] = None,
                         iq_per_site: Optional[Dict[Site, ImageQuality]] = None):

        nightly_timeline = NightlyTimeline()

        for night_idx in sorted(night_indices):
            night_indices = np.array([night_idx])

            for site in sites:
                # Reset the Selector to the default weather for the night.
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
                    match event:
                        case EveningTwilight(new_night_start, _, _):
                            if night_start is not None:
                                raise ValueError(f'Multiple evening twilight events for night index {night_idx} '
                                                 f'at site {site.name}: was {night_start}, now {new_night_start}.')
                            night_start = new_night_start

                        case MorningTwilight():
                            # This just marks the end of the observing night and triggers the time accounting.
                            if night_start is None:
                                raise ValueError(f'Morning twilight event for night index {night_idx} '
                                                 f'at site {site.name} before evening twilight event.')
                            night_start = None
                            night_done = True

                        case WeatherChange(_, _, affected_site, new_conditions):
                            if night_start is None:
                                raise ValueError(f'Event for night index {night_idx} at site {site.name} occurred '
                                                 f'before twilight: {event}.')
                            selector.update_conditions(affected_site, new_conditions)

                        case _:
                            raise NotImplementedError(f'Received unsupported event: {event.__class__.__name__}')

                    # Calculate the time slot of the event. Note that if the night is done, it is None.
                    if night_done:
                        event_start_time_slot = None
                        end_timeslot_bounds = None
                    else:
                        event_start_time_slot = event.to_timeslot_idx(night_start,
                                                                      collector.time_slot_length.to_datetime())
                        end_timeslot_bounds = {site: TimeslotIndex(event_start_time_slot)}

                        # If tbe following conditions are met:
                        # 1. there are plans (i.e. plans have been generated by this point, meaning at least the
                        #    evening twilight event has occurred); and
                        # 2. a new plan is to be produced (TODO: GSCHED-515)
                        # then perform time accounting.
                        #
                        # This will also perform the final time accounting when the night is done and the
                        # morning twilight event has occurred.
                    if plans is not None:
                        collector.time_accounting(plans,
                                                  sites=frozenset({site}),
                                                  end_timeslot_bounds=end_timeslot_bounds)

                        # If the following conditions are met:
                        # 1. the night is not done;
                        # 2. a new plan is to be produced (TODO: GSCHED-515)
                        # fetch a new selection and produce a new plan.
                    if not night_done:
                        selection = selector.select(night_indices=night_indices,
                                                    sites=frozenset([event.site]),
                                                    starting_time_slots={site: {night_idx: event_start_time_slot
                                                                                for night_idx in night_indices}})

                        # Right now the optimizer generates List[Plans], a list of plans indexed by
                        # every night in the selection. We only want the first one, which corresponds
                        # to the current night index we are looping over.
                        plans = optimizer.schedule(selection)[0]
                        nightly_timeline.add(NightIndex(night_idx),
                                             site,
                                             TimeslotIndex(event_start_time_slot),
                                             event,
                                             plans[site])

        return nightly_timeline

    def run(self,
            mode: SchedulerModes,
            start_vis: Time,
            end_vis: Time,
            num_nights_to_schedule: int,
            sites: FrozenSet[Site],
            cc_per_site: Optional[Dict[Site, CloudCover]] = None,
            iq_per_site: Optional[Dict[Site, ImageQuality]] = None):

        night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))
        semesters = frozenset([Semester.find_semester_from_date(start_vis.to_value('datetime')),
                               Semester.find_semester_from_date(end_vis.to_value('datetime'))])

        builder = self._setup(night_indices, sites, mode)

        # Build
        collector = builder.build_collector(start_vis,
                                            end_vis,
                                            sites,
                                            semesters,
                                            Blueprints.collector)

        selector = builder.build_selector(collector,
                                          num_nights_to_schedule,
                                          cc_per_site=cc_per_site,
                                          iq_per_site=iq_per_site
                                          )
        optimizer = builder.build_optimizer(Blueprints.optimizer)

        # Add events for twilight
        for site in sites:
            night_events = collector.get_night_events(site)
            for night_idx in night_indices:
                eve_twilight_time = night_events.twilight_evening_12[night_idx].to_datetime()
                eve_twilight = EveningTwilight(start=eve_twilight_time, reason='Evening 12° Twilight', site=site)
                builder.events.add_event(night_idx, site, eve_twilight)

                # Add one time slot to the morning twilight to make sure time accounting is done for entire night.
                morn_twilight_time = night_events.twilight_morning_12[night_idx].to_datetime()
                morn_twilight = MorningTwilight(start=morn_twilight_time, reason='Morning 12° Twilight', site=site)
                builder.events.add_event(night_idx, site, morn_twilight)

        timelines = self._schedule_nights(night_indices,
                                          sites,
                                          collector,
                                          selector,
                                          optimizer,
                                          builder.events,
                                          cc_per_site,
                                          iq_per_site)

        # Calculate plans stats
        plan_summary = StatCalculator.calculate_timeline_stats(timelines, night_indices, sites, collector)

        return timelines, plan_summary
