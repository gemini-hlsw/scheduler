# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
from time import time
import traceback
from typing import Callable, Dict, Optional

from .params import SchedulerParameters, build_params_store
from .plan_worker import SchedulerComputePayload, worker_build, worker_compute
from scheduler.config import config
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

from ..shared_queue import plan_response_subscribers

_logger = logger_factory.create_logger(__name__)


class EngineRT:

    def __init__(
        self,
        params: SchedulerParameters,
        scheduler_queue: SchedulerQueue,
        process_id: str,
        weather_source: WeatherEventSource,
        executor_factory: Optional[Callable[[], Executor]] = None
    ):
        """
        Initializes the EngineRT with the given parameters.

        Args:
            params (SchedulerParameters): Parameters for the scheduler.
            scheduler_queue (SchedulerQueue): Queue for the scheduler.
            process_id (str): Unique process ID from SchedulerProcess
            executor_factory (Callable, optional): Creates the plan-worker
                executor (also after a recycle); defaults to a single-worker
                process pool.
        """
        _logger.debug("Initializing real-time engine...")
        self.params = params
        self.scheduler_queue = scheduler_queue
        self.process_id = process_id
        self.weather_source = weather_source
        self.start_time = time()
        self._executor_factory = executor_factory or (lambda: ProcessPoolExecutor(max_workers=1))
        self._executor: Optional[Executor] = None

    def _get_executor(self) -> Executor:
        """The worker executor, created on first use (and after a recycle).

        A single warm worker owns the SCP (built via worker_build); the main
        process only ships picklable payloads, so CPU-bound plan computation
        never blocks the event loop.
        """
        if self._executor is None:
            self._executor = self._executor_factory()
        return self._executor

    def shutdown_workers(self) -> None:
        """Shut down the worker pool. Safe to call when nothing was started.

        A hung worker never finishes its task, so shutdown() alone would leave
        the OS process alive; kill the pool's processes explicitly (private
        attribute, but there is no public kill on ProcessPoolExecutor).
        """
        if self._executor is not None:
            processes = dict(getattr(self._executor, "_processes", None) or {})
            self._executor.shutdown(wait=False, cancel_futures=True)
            for process in processes.values():
                process.kill()
            self._executor = None

    @staticmethod
    def _plan_timeout() -> float:
        """Seconds one plan (worker build + compute) may take.

        Config key operation.plan_timeout_seconds; the PLAN_TIMEOUT_SECONDS
        env var is resolved on every read, so it can be changed dynamically.
        """
        return float(config.operation.plan_timeout_seconds)

    async def _fetch_variants(self) -> Dict[Site, VariantSnapshot]:
        """Current weather variant per site, fetched in the parent process
        (async GPP call) and shipped to the worker in the compute payload."""
        _logger.info("Fetching current weather variants...")
        current_state = await self.weather_source.get_current_state()
        variants = {}
        for site_state in current_state:
            variant = VariantSnapshot(iq=ImageQuality(site_state["imageQuality"]),
                                      cc=CloudCover(site_state["cloudCover"]),
                                      wind_dir=Angle(site_state["windDirection"], unit=u.deg),
                                      wind_spd=site_state["windSpeed"] * (u.m / u.s))
            _logger.info(f"Variant for site {site_state['site']} is {variant}")
            variants[Site[site_state["site"]]] = variant
        return variants

    async def compute_event_plan(self, event: Event):
        """
        Compute a new plan for the given event, gated by the aggregator interlock.

        Hard interlock: we never create a plan while the visibility aggregator is
        running, blocking until it finishes (it only starts when no night is being
        executed). We then publish that a plan computation is in progress so a
        cron tick won't begin aggregating concurrently, and clear it when done.

        The computation itself runs in the worker process: the parent gathers
        the async inputs (build params, weather variants) and ships picklable
        commands, so the event loop stays responsive.

        Args:
            event (Event): The event to compute the plan for.
        Returns:
            NightPlansWithEvent: The new plan for the event.
        """
        await coordination.wait_until_aggregator_idle()
        await coordination.signal_plan_in_progress(
            holder=self.process_id,
            detail={"event": str(event.description)},
        )
        try:
            build_params = await build_params_store.get()
            variants = await self._fetch_variants()

            timeout = self._plan_timeout()
            try:
                plans = await asyncio.wait_for(
                    self._run_plan_in_worker(event, build_params, variants),
                    timeout=timeout,
                )
            except TimeoutError:
                _logger.error(f"Plan computation for '{event.description}' exceeded the "
                              f"{timeout:.0f}s budget; recycling the worker pool.")
                self.shutdown_workers()
                raise TimeoutError(f"Plan computation exceeded the {timeout:.0f}s budget; "
                                   f"the worker pool was recycled.") from None

            splans = SPlans.from_computed_plans(plans, self.params.sites)

            return NightPlansWithEvent(night_plans=splans,
                                       event=f"{event.description} @{event.time if event.time else 'Start of Night'}")
        finally:
            await coordination.signal_plan_done()

    async def _run_plan_in_worker(self, event: Event, build_params, variants):
        """Ship the build command and compute payload to the worker process."""
        loop = asyncio.get_running_loop()
        executor = self._get_executor()

        await loop.run_in_executor(executor, worker_build, self.params, build_params)

        payload = SchedulerComputePayload(event=event,
                                          sites=self.params.sites,
                                          night_times=build_params.get_night_times(),
                                          variants=variants)
        return await loop.run_in_executor(executor, worker_compute, payload)

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
