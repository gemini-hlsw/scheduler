# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from time import time
import traceback
from .params import SchedulerParameters, build_params_store
from .plan_computation import compute_event_plans
from scheduler.core.scp.scp import SCP
from scheduler.core.builder.modes import dispatch_with
from scheduler.core.builder import Blueprints, SimulationBuilder
from scheduler.core.sources import Sources
from scheduler.services import logger_factory
from scheduler.core.events.queue.events import Event, EndOfNightEvent
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue
from scheduler.graphql_mid.types import NightPlansError
from scheduler.graphql_mid.types import SPlans, NightPlansWithEvent
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
        weather_source: WeatherEventSource
    ):
        """
        Initializes the EngineRT with the given parameters.
        
        Args:
            params (SchedulerParameters): Parameters for the scheduler.
            scheduler_queue (SchedulerQueue): Queue for the scheduler.
            process_id (str): Unique process ID from SchedulerProcess
        """
        _logger.debug("Initializing real-time engine...")
        self.params = params
        self.scheduler_queue = scheduler_queue
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

    async def _compute_event_plan(self, event: Event):
        """
        Compute a new plan based on the given event.

        Args:
            event (Event): The event to compute the plan for.
        Returns:
            NewPlansRT: The new plan for the event.
        """
        await self.build()
        await self.init_variant()

        build_params = await build_params_store.get()
        night_times = build_params.get_night_times()

        # Pure sync computation (RT-22); runs in a worker process from RT-25 on.
        plans = compute_event_plans(self.scp, self.params.sites, event, night_times)

        splans = SPlans.from_computed_plans(plans, self.params.sites)

        return NightPlansWithEvent(night_plans=splans, event=f"{event.description} @{event.time if event.time else 'Start of Night'}")

    async def run(self):
        """
        Run the EngineRT process throughout the set of nights.
        """
        try:
            # Run event loop while still in the same night
            while True:
                try:
                    # Wait for the next event
                    event, plan = await self.scheduler_queue.consume_events(self.compute_event_plan)
                    _logger.debug(f"Received scheduler event: {event}")

                    # Check if we have reached the end of the night
                    if isinstance(event, EndOfNightEvent):
                        _logger.info("Night end event received, ending night scheduling loop.")
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
