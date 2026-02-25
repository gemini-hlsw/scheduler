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
        self.night_monitor = NightMonitor(current_night, self.params.sites, self.scheduler_queue)

        # Start night monitor
        await self.night_monitor.start()
        _logger.info("Night monitor started.")

        # Get the weather source gql client
        weather_source = self.night_monitor.get_weather_source()

        # Initialize Real Time Engine
        self.engine = EngineRT(self.params, self.scheduler_queue, self.process_id, weather_source=weather_source)

        # Initialize the engine variants
        self._engine_task = asyncio.create_task(self.engine.run())
        _logger.info("Engine started.")

        self.running_event.set()