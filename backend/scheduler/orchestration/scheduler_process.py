# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from datetime import timedelta

from astropy.time import Time

from scheduler.core.events.queue.nightly_timeline_store import NightlyTimelineStore
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue
from scheduler.graphql_mid.types import NightPlansError
from scheduler.night_monitor.night_monitor import NightMonitor
from scheduler.services.logger_factory import create_logger
from scheduler.shared_queue import plan_response_subscribers
from scheduler.engine import SchedulerParameters
from scheduler.engine import EngineRT

_logger = create_logger(__name__, with_id=False)

__all__ = ["SchedulerProcess"]

class SchedulerProcess:
    """
    Class to manage a scheduler process should be able to start, stop, 
    and monitor a scheduler process asynchronously.
    Should be able to run through multiple nights until stopped by the UI
    or reached the specified number of nights or date.
    """

    def __init__(self,
                 process_id: str,
                 params: SchedulerParameters):
        """
        Initialize the scheduler process
        
        Args:
            process_id (str): A unique identifier for the scheduler process.
            params (SchedulerParameters): The parameters for the scheduler process.
        """

        self.process_id = process_id
        self.scheduler_queue = SchedulerQueue()
        self.nightly_timeline_store = NightlyTimelineStore()
        self.params = params
        self.running_event = asyncio.Event()
        self.engine = None
        self.night_monitor = None
        self.task = None
        self._engine_task = None

    async def stop_process(self):
        """
        Stop the scheduler process, releasing every owned task and subscription.
        """

        _logger.info("Stopping scheduler process...")
        self.running_event.clear()

        # Stop event production first so nothing new reaches the engine.
        if self.night_monitor is not None:
            await self.night_monitor.shutdown()

        for task in (self._engine_task, self.task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # The task already died on its own; stopping must not fail.
                _logger.exception(f"Task {task.get_name()} failed before stop_process.")

        self._engine_task = None
        self.task = None

    async def start_task(self):
        """
        Start the scheduler process as an asyncio task
        """
        self.task = asyncio.create_task(self.run())

    def is_running(self) -> bool:
        """
        Check if the scheduler process is running
        """

        return self.running_event.is_set()

    async def run(self):
        """
        Run the scheduler process
        """

        _logger.info("Start running scheduler process...")


        night_index = 0
        current_night = self.params.start + timedelta(days=night_index)
        # Initialize the night monitor
        self.night_monitor = NightMonitor(current_night,
                                          self.params.sites,
                                          self.scheduler_queue,
                                          self.nightly_timeline_store)

        # Start night monitor
        await self.night_monitor.start()
        _logger.info("Night monitor started.")

        # Get the weather source gql client
        weather_source = self.night_monitor.get_weather_source()

        # Initialize Real Time Engine
        self.engine = EngineRT(self.params,
                               self.scheduler_queue,
                               self.process_id,
                               weather_source=weather_source,
                               nightly_timeline_store=self.nightly_timeline_store)

        # Initialize the engine variants
        self._engine_task = asyncio.create_task(self.engine.run())
        self._engine_task.add_done_callback(self._on_engine_done)
        _logger.info("Engine started.")

        self.running_event.set()

    def _on_engine_done(self, task: asyncio.Task) -> None:
        """
        Supervise the engine task. Logs when an exception happens
        and cleans when is done.
        """
        if task.cancelled():
            # Normal shutdown path (stop_process), nothing to report.
            return

        exc = task.exception()
        if exc is None:
            _logger.info(f"Engine task for process '{self.process_id}' finished.")
            self.running_event.clear()
            return

        _logger.error(f"Engine task for process '{self.process_id}' crashed.",
                      exc_info=exc)
        self.running_event.clear()
        # asyncio.Queue is unbounded, so put_nowait cannot raise QueueFull.
        for q in plan_response_subscribers.get(self.process_id, set()):
            q.put_nowait(NightPlansError(error=str(exc)))
