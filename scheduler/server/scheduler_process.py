# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from datetime import timedelta

from astropy.time import Time

from scheduler.clients import SchedulerQueue
from scheduler.night_monitor.night_monitor import NightMonitor
from scheduler.services.logger_factory import create_logger
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
        self.params = params
        self.running_event = asyncio.Event()
        self.engine = None
        self.night_monitor = None
        self._engine_task = None

    async def stop_process(self):
        """
        Stop the scheduler process
        """

        _logger.info("Stopping scheduler process...")
        self.running_event.clear()
        # Cancel the running task if needed
        if hasattr(self, 'task'):
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def start_task(self):
        """
        Start the scheduler process as an asyncio task
        """

        self.task = asyncio.create_task(self.run())
        # await self.task

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
        night_monitor = NightMonitor(current_night, self.params.sites, self.scheduler_queue)

        # Get initial states
        (initial_resource, initial_weather) = await night_monitor.get_initial_state()

        # Start night monitor
        await night_monitor.start()
        _logger.info("Night monitor started.")

        # Initialize Real Time Engine
        engine = EngineRT(self.params, self.scheduler_queue, self.process_id)
        await engine.build()

        # Initialize the engine variants
        engine.init_variant(initial_weather)

        self._engine_task = asyncio.create_task(engine.run())
        _logger.info("Engine started.")

        self.running_event.set()

        # Loop through nights until stopped or reached the specified number of nights/date

        #while self.running_event.is_set():
        #    # Check if we have reached the end date or number of nights
        #    if (self.params.end and current_night >= self.params.end) or \
        #         (self.params.num_nights_to_schedule and night_index >= self.params.num_nights_to_schedule):
        #        _logger.info("End date reached, ending scheduler process.")
        #        return

            # RT should not do more than one night
            # night_index += 1
            #current_night = self.params.start + timedelta(days=night_index)
            #_logger.debug(f"Next night: {current_night}")

    async def update_params(self, params: SchedulerParameters, night_start: Time, night_end: Time):

        self.params = params
        # rebuild engine
        self._engine_task.cancel()
        self._engine_task = None
        self.engine = EngineRT(
            self.params,
            self.scheduler_queue,
            self.process_id,
            night_start_time=night_start,
            night_end_time=night_end,
        )
        # await self.engine.build()
        # This shouldnt be needed if we are getting the initial value from the weather service
        # self.engine.init_variant()
        self._engine_task = asyncio.create_task(self.engine.run())