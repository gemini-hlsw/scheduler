# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from datetime import datetime

from typing import FrozenSet, Optional, List, Dict

import numpy as np
from astropy.time import Time
from lucupy.minimodel import Site, Semester, NightIndex, TimeslotIndex, CloudCover, ImageQuality

from scheduler.core.builder.modes import dispatch_with, SchedulerModes
from scheduler.core.components.collector import Collector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.components.selector import Selector
from scheduler.core.eventsqueue import EventQueue, EveningTwilight, MorningTwilight, WeatherChange
from scheduler.core.eventsqueue.nightchanges import NightTimeline
from scheduler.core.plans import Plans
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.builder import SchedulerBuilder, Blueprints
from scheduler.core.sources import Sources
from scheduler.core.statscalculator import StatCalculator
from scheduler.db.planmanager import PlanManager

from definitions import ROOT_DIR


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

        night_timeline = NightTimeline({night_index: {site: [] for site in sites}
                                        for night_index in night_indices})

        for night_idx in sorted(night_indices):
            night_indices = np.array([night_idx])

            for site in sites:

                # Reset the Selector to the default weather for the night.
                cc_value = cc_per_site and cc_per_site.get(site)
                iq_value = iq_per_site and iq_per_site.get(site)
                selector.update_cc_and_iq(site, cc_value, iq_value)

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
                            event_start_time_slot = event.to_timeslot_idx(night_start,
                                                                          collector.time_slot_length.to_datetime())
                            plan = night_timeline.get_final_plan(night_idx, site)
                            night_timeline.add(NightIndex(night_idx),
                                               site,
                                               TimeslotIndex(event_start_time_slot),
                                               event,
                                               plan)

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

        timelines = self._schedule_nights(night_indices,
                                          sites,
                                          collector,
                                          selector,
                                          optimizer,
                                          builder.events,
                                          cc_per_site,
                                          iq_per_site)
        plans = []
        for n_idx in night_indices:
            p = Plans(collector.night_events, n_idx)
            for site in sites:
                plan = timelines.get_plan_by_event(n_idx, site, MorningTwilight)
                if plan:
                    p[site] = plan
                else:
                    raise ValueError(f'Plan not found for site {site} at night {n_idx}')
            plans.append(p)

        # Calculate plans stats
        plan_summary = StatCalculator.calculate_timeline_stats(timelines, night_indices, sites, collector)

        return timelines, plan_summary
