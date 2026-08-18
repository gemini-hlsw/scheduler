# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import datetime
from time import time
import traceback
import numpy as np
from .params import SchedulerParameters, build_params_store
from scheduler.core.scp.scp import SCP
from scheduler.core.builder.modes import dispatch_with
from scheduler.core.builder import Blueprints, SimulationBuilder
from scheduler.core.sources import Sources
from scheduler.core.plans import NightStats
from scheduler.services import logger_factory
from scheduler.core.events.queue import NightlyTimelineStore
from scheduler.core.events.queue.events import CustomStartOfNightEvent, Event, EndOfNightEvent
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue
from scheduler.core.statscalculator import StatCalculator
from scheduler.graphql_mid.types import NewNightPlans, SNightTimelines, SRunSummary
from scheduler.graphql_mid.types import NightPlansError, SPlans
from scheduler.graphql_mid.types import NightPlansWithEvent
from scheduler.night_monitor.event_sources import WeatherEventSource
from scheduler.services.visibility_aggregator import coordination
from lucupy.minimodel import VariantSnapshot, ImageQuality, CloudCover, Site
from astropy.coordinates import Angle
from astropy import units as u


__all__ = [
    'EngineRT'
]

from ..core.components.ranker import DefaultRanker
from ..shared_queue import plan_response_subscribers

_logger = logger_factory.create_logger(__name__)


class EngineRT:

    def __init__(
        self,
        params: SchedulerParameters,
        scheduler_queue: SchedulerQueue,
        process_id: str,
        weather_source: WeatherEventSource,
        nightly_timeline_store: NightlyTimelineStore
    ):
        """
        Initializes the EngineRT with the given parameters.

        Args:
            params (SchedulerParameters): Parameters for the scheduler.
            scheduler_queue (SchedulerQueue): Queue for the scheduler.
            process_id (str): Unique process ID from SchedulerProcess
            weather_source (WeatherEventSource): Source for the current weather state.
            nightly_timeline_store (NightlyTimelineStore): Shared store for the nightly timeline,
                read by the Night Monitor to know the plan currently in effect.
        """
        _logger.debug("Initializing real-time engine...")
        self.params = params
        self.scheduler_queue = scheduler_queue
        self.nightly_timeline_store = nightly_timeline_store
        self.scp = None
        self.process_id = process_id
        self.weather_source = weather_source
        self.sources = Sources()
        self.start_time = time()

    async def build(self) -> None:
        """
        Creates a Scheduler Core Pipeline based on the parameters.
        Also initialize both the Event Queue , both needed for the scheduling process.
        """
        # Create builder based in the mode to create SCP
        builder = dispatch_with(self.sources, None)
        if not isinstance(builder, SimulationBuilder):
            raise RuntimeError("Builder must be Simulation to use async build method.")

        build_params = await build_params_store.get()
        night_times = build_params.get_night_times()
        _logger.info(
            f"Build params: vis_start={build_params.visibility_start}, vis_end={build_params.visibility_end}, night_times={night_times}")

        vis_start = build_params.visibility_start or self.params.start
        vis_end = build_params.visibility_end or self.params.end_vis
        programs_list = build_params.program_list or self.params.programs_list

        collector = await builder.async_build_collector(
            start=vis_start,
            end=vis_end,
            num_of_nights=self.params.num_nights_to_schedule,
            sites=self.params.sites,
            semesters=self.params.semesters,
            blueprint=Blueprints.collector,
            night_times=night_times,
            program_list=programs_list
        )


        selector = builder.build_selector(collector=collector,
                                          num_nights_to_schedule=self.params.num_nights_to_schedule,
                                          blueprint=Blueprints.selector)

        optimizer = builder.build_optimizer(Blueprints.optimizer)
        ranker = DefaultRanker(collector,
                               self.params.night_indices,
                               self.params.sites,
                               params=self.params.ranker_parameters)

        self.scp = SCP(collector, selector, optimizer, ranker)
        _logger.info("SCP successfully built.")

    async def init_variant(self) -> None:
        """
        Initialize site variants with default values.
        If a new variant is presented via event, set to those.
        """

        _logger.info("Updating initial variants...")
        current_state = await self.weather_source.get_current_state()
        for site_state in current_state:
            initial_variant = VariantSnapshot(iq=ImageQuality(site_state["imageQuality"]),
                                          cc=CloudCover(site_state["cloudCover"]),
                                          wind_dir=Angle(site_state["windDirection"], unit=u.deg),
                                          wind_spd=site_state["windSpeed"] * (u.m / u.s))

            _logger.info(f"Initial variant for site {site_state['site']} is {initial_variant}")
            self.scp.selector.update_site_variant(Site[site_state["site"]], initial_variant)
        _logger.info("Initial weather variants successfully updated.")

    async def compute_event_plan(self, event: Event):
        """
        Compute a new plan for the given event, gated by the aggregator interlock.

        Hard interlock: we never create a plan while the visibility aggregator is
        running, blocking until it finishes (it only starts when no night is being
        executed). We then publish that a plan computation is in progress so a
        cron tick won't begin aggregating concurrently, and clear it when done.
        """
        await coordination.wait_until_aggregator_idle()
        await coordination.signal_plan_in_progress(
            holder=self.process_id,
            detail={"event": str(event.description)},
        )
        try:
            return await self._compute_event_plan(event)
        finally:
            await coordination.signal_plan_done()

    async def _compute_event_start_timeslot(self, event: Event):
        """
        Compute the start timeslot for the given event.
        The start timeslot for each site should be:
        - The event time if happens after the twilight and after the custom start
        - The custom start if defined and happens after the twilight
        - The twilight timeslot

        Args:
            event (Event): The event to compute the start timeslot for.

        Returns:
            Timeslot: The start timeslot for the event.
        """
        start_timeslot = {}
        async with self.nightly_timeline_store.mutate() as nightly_timeline:
            for site in self.params.sites:
                night_start, night_end = self.scp.collector.get_night_length(site, 0)
                nightly_timeline.set_night_length(0, site, night_start, night_end)

                event_timeslot = event.to_timeslot_idx(
                    night_start,
                    self.scp.collector.time_slot_length.to_datetime()
                )

                _logger.info(f"Computing plan for site {site.name} starting on timeslot: {event_timeslot if event_timeslot > 0 else 0}, "
                             f"night_start={night_start}, event_time={event.time}")
                start_timeslot[site] = {np.int64(0): event_timeslot if event_timeslot > 0 else 0}

        return start_timeslot

    async def _compute_event_plan(self, event: Event):
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
        # TODO: Specific logic for events
        # In theory this should be a shared process for all events.
        # Meaning the process of setup the SCP and run a schedule is independent from the type of event.
        # sites_to_update = self.params.sites
        # if 'Weather' in event.trigger_event:
        #     self.scp.selector.update_site_variant(event.site, event.event.variant_change)
        #     sites_to_update = [event.site]
        await self.init_variant()

        start_timeslot = await self._compute_event_start_timeslot(event)

        plans = self.scp.run_rt(start_timeslot)

        # The start timeslot section above already released the lock, so the Night Monitor
        # readers are only blocked while the timeline is actually being updated.
        async with self.nightly_timeline_store.mutate() as nightly_timeline:
            for site in self.params.sites:
                nightly_timeline.add(plans.night_idx, site, start_timeslot[site][0], event, plans.plans[site])
                nightly_timeline.calculate_time_losses(plans.night_idx, site)
            run_summary = StatCalculator.calculate_stitched_timeline_stats(nightly_timeline,
                                                                          self.params.night_indices,
                                                                          self.params.sites,
                                                                          self.scp.collector,
                                                                          self.scp.ranker)
            s_timelines = SNightTimelines.from_computed_stitched_timelines(nightly_timeline)
        s_plan_summary = SRunSummary.from_computed_run_summary(run_summary)
        return NewNightPlans(night_plans=s_timelines, plans_summary=s_plan_summary)

    async def run(self):
        """
        Run the EngineRT process throughout the set of nights.
        """
        try:
            # Run the event loop while still in the same night
            while True:
                try:
                    # Wait for the next event
                    event, plan = await self.scheduler_queue.consume_events(self.compute_event_plan)
                    _logger.debug(f"Received scheduler event: {event}")

                    # Check if we have reached the end of the night
                    if isinstance(event, EndOfNightEvent):
                        _logger.info("Night end event received, ending night scheduling loop.")
                        # The night is over, so its timeline must not be served as the plan in
                        # effect to whatever runs next.
                        await self.nightly_timeline_store.reset()
                        break
                    # Plan is already computed by the callback in consume_events
                    for q in plan_response_subscribers.get(self.process_id, set()):
                        await q.put(plan)

                except Exception as e:
                    traceback.print_exc()
                    _logger.error(f"Error in scheduler process: {e}")
                    for q in plan_response_subscribers.get(self.process_id, set()):
                        await q.put(NightPlansError(error=str(e)))

        except asyncio.CancelledError:
            _logger.info("Scheduler process was cancelled.")

        except Exception as e:
            _logger.error(f"Error in scheduler process: {e}")
            raise
