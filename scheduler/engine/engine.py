# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Tuple

from astropy.time import Time

from lucupy.timeutils import time2slots

from .params import SchedulerParameters
from scheduler.core.scp.scp import SCP

from scheduler.core.builder.modes import dispatch_with
from scheduler.core.builder import Blueprints
from scheduler.core.events.queue import EventQueue, EveningTwilightEvent, WeatherChangeEvent, MorningTwilightEvent
from scheduler.core.events.queue import NightlyTimeline
from scheduler.core.sources import Sources
from scheduler.core.statscalculator import StatCalculator
from scheduler.services import logger_factory

from time import time


__all__ = [
    'Engine'
]

from ..core.components.ranker import DefaultRanker

from ..core.events.cycle.cycle import EventCycle

from ..core.statscalculator.run_summary import RunSummary

_logger = logger_factory.create_logger(__name__)


class Engine:

    def __init__(self, params: SchedulerParameters, night_start_time: Time | None = None, night_end_time: Time | None = None):
        self.params = params
        self.sources = Sources()
        self.change_monitor = None
        self.start_time = time()
        self.night_start_time = night_start_time
        self.night_end_time = night_end_time

    def build(self) -> SCP:
        """
        Creates a Scheduler Core Pipeline based on the parameters.
        Also initialize both the Event Queue , both needed for the scheduling process.
        """

        # Create event queue to handle incoming events.
        self.queue = EventQueue(self.params.night_indices, self.params.sites)

        # Create builder based in the mode to create SCP
        builder = dispatch_with(self.sources, self.queue)

        collector = builder.build_collector(start=self.params.start,
                                            end=self.params.end_vis,
                                            num_of_nights=self.params.num_nights_to_schedule,
                                            sites=self.params.sites,
                                            semesters=self.params.semesters,
                                            blueprint=Blueprints.collector,
                                            night_start_time=self.night_start_time,
                                            night_end_time=self.night_end_time,
                                            program_list=self.params.programs_list)

        selector = builder.build_selector(collector=collector,
                                          num_nights_to_schedule=self.params.num_nights_to_schedule,
                                          blueprint=Blueprints.selector)

        optimizer = builder.build_optimizer(Blueprints.optimizer)
        ranker = DefaultRanker(collector,
                               self.params.night_indices,
                               self.params.sites,
                               params=self.params.ranker_parameters)

        return SCP(collector, selector, optimizer, ranker)

    def _setup(self, scp: SCP, queue: EventQueue) -> None:
        """
        This process is needed before the scheduling process can occur.
        It handles the initial weather conditions, the setup for both twilights,
        the fault handling and other events to be added to the queue.
        Returns the initial weather variations for each site and for each night.
        """
        # TODO: The weather process might want to be done separately from the fulfillment of the queue.
        # TODO: specially since those process in the PRODUCTION mode are going to be different.

        sites = self.params.sites
        night_indices = self.params.night_indices
        time_slot_length = scp.collector.time_slot_length.to_datetime()

        # Add the twilight events for every night at each site.
        # The morning twilight will force time accounting to be done on the last generated plan for the night.
        for site in sites:
            night_events = scp.collector.get_night_events(site)
            for night_idx in night_indices:
                # this would be probably because when the last time the resource pickle was created, it was winter time
                # or different.
                eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
                eve_twi = EveningTwilightEvent(site=site, time=eve_twi_time, description='Evening 12° Twilight')
                queue.add_event(night_idx, site, eve_twi)

                # Get the weather events for the site for the given night date.
                night_date = eve_twi_time.date()
                morn_twi_time = night_events.twilight_morning_12[night_idx].to_datetime(
                    site.timezone) - time_slot_length
                # morn_twi_slot = time2slots(time_slot_length, morn_twi_time - eve_twi_time)
                morn_twi_slot = night_events.num_timeslots_per_night[night_idx]

                # Get the weather events for the site for the given night date.
                # Get the VariantSnapshots for the times of the night where the variant changes.
                variant_changes_dict = scp.collector.sources.origin.env.get_variant_changes_for_night(site, night_date)
                for variant_datetime, variant_snapshot in variant_changes_dict.items():
                    variant_timeslot = time2slots(time_slot_length, variant_datetime - eve_twi_time)

                    # If the variant happens before or at the first time slot, we set the initial variant for the night.
                    # The closer to the first time slot, the more accurate, and the ordering on them will overwrite
                    # the previous values.
                    if variant_timeslot <= 0:
                        _logger.debug(f'WeatherChange for site {site.name}, night {night_idx}, occurs before '
                                      '0: ignoring.')
                        continue

                    if variant_timeslot >= morn_twi_slot:
                        _logger.debug(f'WeatherChange for site {site.name}, night {night_idx}, occurs after '
                                      f'{morn_twi_slot}: ignoring.')
                        continue

                    variant_datetime_str = variant_datetime.strftime('%Y-%m-%d %H:%M')
                    weather_change_description = (f'Weather change at {site.name}, {variant_datetime_str}: '
                                                  f'IQ -> {variant_snapshot.iq.name}, '
                                                  f'CC -> {variant_snapshot.cc.name}')
                    weather_change_event = WeatherChangeEvent(site=site,
                                                              time=variant_datetime,
                                                              description=weather_change_description,
                                                              variant_change=variant_snapshot)
                    queue.add_event(night_idx, site, weather_change_event)

                # Process the unexpected closures for the night at the site -> Weather loss events
                closure_set = scp.collector.sources.origin.resource.get_unexpected_closures(site, night_date)
                for closure in closure_set:
                    closure_start, closure_end = closure.to_events()
                    queue.add_event(night_idx, site, closure_start)
                    queue.add_event(night_idx, site, closure_end)

                # Process the fault reports for the night at the site.
                faults_set = scp.collector.sources.origin.resource.get_faults(site, night_date)
                for fault in faults_set:
                    fault_start, fault_end = fault.to_events()
                    queue.add_event(night_idx, site, fault_start)
                    queue.add_event(night_idx, site, fault_end)

                # Process the ToO activation for the night at the site.

                too_set = scp.collector.sources.origin.resource.get_toos(site, night_date)
                for too in too_set:
                    too_event = too.to_event()
                    queue.add_event(night_idx, site, too_event)

                morn_twi = MorningTwilightEvent(site=site, time=morn_twi_time, description='Morning 12° Twilight')
                queue.add_event(night_idx, site, morn_twi)

                # TODO: If any InterruptionEvents occur before twilight, block the site with the event.

    def schedule(self) -> Tuple[RunSummary, NightlyTimeline]:

        nightly_timeline = NightlyTimeline()
        scp = self.build()
        queue = EventQueue(self.params.night_indices, self.params.sites)
        self._setup(scp, queue)
        event_cycle = EventCycle(self.params, queue, scp)
        # tn0 = time()
        for night_idx in sorted(self.params.night_indices):
            # print(f'Engine: starting night {night_idx + 1}: {scp.collector.time_grid[night_idx]}')
            for site in sorted(self.params.sites, key=lambda site: site.name):
                event_cycle.run(site, night_idx, nightly_timeline)
                nightly_timeline.calculate_time_losses(night_idx, site)
            # tn1 = time()
            # print(f'Night {night_idx + 1} scheduled in {(tn1 - tn0) / 60.} min')
            # nightly_timeline.display(night_idx_sel=night_idx)
            # tn0 = tn1

        # TODO: Add plan summary to nightlyTimeline
        run_summary = StatCalculator.calculate_timeline_stats(nightly_timeline,
                                                              self.params.night_indices,
                                                              self.params.sites,
                                                              scp.collector,
                                                              scp.ranker)
        
        _logger.info(f'Plan calculated in {time() - self.start_time}')

        return run_summary, nightly_timeline
