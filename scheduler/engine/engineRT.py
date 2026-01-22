# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import datetime

import numpy as np
from astropy.time import Time

from .params import SchedulerParameters
from scheduler.core.scp.scp import SCP

from scheduler.core.builder.modes import dispatch_with
from scheduler.core.builder import Blueprints, SimulationBuilder
from scheduler.core.sources import Sources, Origin, Origins
from scheduler.core.plans import NightStats
from scheduler.services import logger_factory
from scheduler.core.events.queue.events import EndOfNightEvent
from scheduler.graphql_mid.types import NightPlansError
from scheduler.shared_queue import plan_response_queue
from scheduler.clients.scheduler_queue_client import SchedulerQueue, SchedulerEvent
from scheduler.events import to_timeslot_idx
from scheduler.graphql_mid.types import SPlans, NewPlansRT

from lucupy.minimodel import VariantSnapshot, ImageQuality, CloudCover, Site
from astropy.coordinates import Angle
from astropy import units as u


from time import time


__all__ = [
    'EngineRT'
]

from ..core.components.ranker import DefaultRanker
from ..core.output import print_collector_info
from ..core.statscalculator import StatCalculator

_logger = logger_factory.create_logger(__name__)


class EngineRT:

    def __init__(
        self,
        params: SchedulerParameters,
        scheduler_queue: SchedulerQueue,
        process_id: str,
        night_start_time: Time | None = None,
        night_end_time: Time | None = None
    ):
        """
        Initializes the EngineRT with the given parameters.
        
        Args:
            params (SchedulerParameters): Parameters for the scheduler.
            scheduler_queue (SchedulerQueue): Queue for the scheduler.
            process_id (str): Unique process ID from SchedulerProcess
            night_start_time (Time | None): Optional start time of the night.
            night_end_time (Time | None): Optional end time of the night.
        """
        _logger.debug("Initializing real-time engine...")
        self.params = params
        self.scheduler_queue = scheduler_queue
        self.process_id = process_id
        self.sources = Sources()
        self.start_time = time()
        self.night_start_time = None
        self.night_end_time = None

    async def build(self) -> SCP:
        """
        Creates a Scheduler Core Pipeline based on the parameters.
        Also initialize both the Event Queue , both needed for the scheduling process.
        """
        # Create builder based in the mode to create SCP
        builder = dispatch_with(self.sources, None)
        if not isinstance(builder, SimulationBuilder):
            raise RuntimeError("Builder must be Simulation to use async build method.")

        collector = await builder.async_build_collector(start=self.params.start,
                                            end=self.params.end_vis,
                                            num_of_nights=self.params.num_nights_to_schedule,
                                            sites=self.params.sites,
                                            semesters=self.params.semesters,
                                            blueprint=Blueprints.collector,
                                            night_start_time=self.night_start_time,
                                            night_end_time=self.night_end_time,
                                            program_list=self.params.programs_list)
        print_collector_info(collector)
        print('GS: ',collector.night_events[Site.GS].twilight_morning_12)
        print('GN: ', collector.night_events[Site.GN].twilight_morning_12)

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

    def init_variant(self):
        """
        Initialize site variants with default values.
        """
        # Should get variants from weather service in the future
        initial_variant = VariantSnapshot(iq=ImageQuality(0.2),
                                          cc=CloudCover(0.5),
                                          wind_dir=Angle(0.0, unit=u.deg),
                                          wind_spd=0.0 * (u.m / u.s))
        for site in self.params.sites:
            self.scp.selector.update_site_variant(site, initial_variant)

        _logger.info("Initial weather variants successfully updated.")

    def compute_event_plan(self, event: SchedulerEvent):
        """
        Compute a new plan based on the given event.
        
        Args:
            event (Event): The event to compute the plan for.
        Returns:
            NewPlansRT: The new plan for the event.
        """
        # Get the timeslots associated with the sites with format
        # {site: {0: current_timeslot}}
        start_timeslot = {}
        for site in self.params.sites:
            night_start_time = self.scp.collector.night_events[site].times[0][0]
            print(event.time, night_start_time)
            event_timeslot = to_timeslot_idx(
                event.time,
                night_start_time.utc.to_datetime(timezone=datetime.timezone.utc),
                self.scp.collector.time_slot_length.to_datetime()
            )
            start_timeslot[site] = {np.int64(0): 0}

        plans = self.scp.run_rt(start_timeslot)

        for site in self.params.sites:
            plans.plans[site].night_stats = NightStats({},0.0,0,{},{})
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

    async def run(self):
        """
        Run the EngineRT process throughout the set of nights.
        """
        try:
            # Run event loop while still in the same night
            while True:
                # Wait for the next event
                event, plan = await self.scheduler_queue.consume_events(self.compute_event_plan)
                _logger.debug(f"Received scheduler event: {event}")

                # Check if we have reached the end of the night
                if isinstance(event, EndOfNightEvent):
                    _logger.info("Night end event received, ending night scheduling loop.")
                    break

                try:
                    # Plan is already computed by the callback in consume_events
                    await plan_response_queue[self.process_id].put(plan)

                except Exception as e:
                    _logger.error(f"Error in scheduler process: {e}")
                    await plan_response_queue[self.process_id].put(NightPlansError(error=str(e)))

        except asyncio.CancelledError:
            _logger.info("Scheduler process was cancelled.")

        except Exception as e:
            _logger.error(f"Error in scheduler process: {e}")
            raise
