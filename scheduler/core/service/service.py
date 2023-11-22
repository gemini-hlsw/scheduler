# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from datetime import datetime

from typing import FrozenSet, Optional

import numpy as np
from astropy.time import Time
from lucupy.minimodel import Site, Semester, NightIndex, TimeslotIndex

from scheduler.core.builder.modes import dispatch_with
from scheduler.core.eventsqueue import EventQueue, EveningTwilight, MorningTwilight, WeatherChange
from scheduler.core.eventsqueue.nightchanges import NightTimeline
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.builder import SchedulerBuilder, Blueprints
from scheduler.core.sources import Sources
from scheduler.core.statscalculator import StatCalculator
from scheduler.db.planmanager import PlanManager

from definitions import ROOT_DIR


class Service:

    @staticmethod
    def _setup(night_indices, sites, mode):

        queue = EventQueue(night_indices, sites)
        sources = Sources()
        builder = dispatch_with(mode, sources, queue)
        return builder

    @staticmethod
    def _schedule_nights(night_indices, sites, collector, selector, optimizer, queue, cc_per_site, iq_per_site):

        night_timeline = NightTimeline({night_index: {site: [] for site in sites}
                                        for night_index in night_indices})

        for night_idx in sorted(night_indices):
            night_indices = np.array([night_idx])

            # Reset the Selector to the default weather for the night.
            for site in sites:
                cc_value = cc_per_site and cc_per_site.get(site)
                iq_value = iq_per_site and iq_per_site.get(site)
                selector.update_cc_and_iq(site, cc_value, iq_value)

            # Run eventless timeline
            selection = selector.select(night_indices=night_indices)
            # Run the optimizer to get the plans for the first night in the selection.
            plans = optimizer.schedule(selection)

            for site in collector.sites:
                # The starting twilight for the night for the site.
                night_start: Optional[datetime] = None
                night_done = False

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

                    # If the night is not done, fetch a new selection given that
                    # the candidates and scores will need to be calculated based on the event.
                    event_start_time_slot = None
                    if not night_done:
                        event_start_time_slot = event.to_timeslot_idx(night_start,
                                                                      collector.time_slot_length.to_datetime())
                        selection = selector.select(night_indices=night_indices,
                                                    sites=frozenset([event.site]),
                                                    starting_time_slots={site: {night_idx: event_start_time_slot
                                                                                for night_idx in night_indices}})

                        # Run the optimizer to get the plans for the first night in the selection.
                        plans = optimizer.schedule(selection)
                        night_timeline.add(NightIndex(night_idx),
                                           site,
                                           TimeslotIndex(event_start_time_slot),
                                           event,
                                           plans[0][site])

                    # Perform the time accounting. If the night is done, we execute it to completion.
                    end_timeslot_bounds = None if night_done else {site: TimeslotIndex(event_start_time_slot)}
                    collector.time_accounting(plans[0],
                                              sites=frozenset({site}),
                                              end_timeslot_bounds=end_timeslot_bounds)
            return night_timeline

    def run(self,
            mode,
            start_vis,
            end_vis,
            num_nights_to_schedule,
            sites,
            cc_per_site,
            iq_per_site):

        night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))
        semesters = frozenset([Semester.find_semester_from_date(start_vis.to_value('datetime')),
                               Semester.find_semester_from_date(end_vis.to_value('datetime'))])

        builder = self._setup(night_indices, sites, mode)

        programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

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

        collector.load_programs(program_provider_class=OcsProgramProvider,
                                data=programs)

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

        plans = self._schedule_nights(night_indices, sites, collector, selector, optimizer, builder.events)
        sorted_plans = [p for _, p in sorted(plans.items())]

        selection = selector.select(sites=sites)

        plans = optimizer.schedule(selection)

        # Calculate plans stats
        plan_summary = StatCalculator.calculate_plans_stats(sorted_plans, collector)
        return plans, plan_summary
