# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from astropy.time import Time, TimeDelta
from scheduler.core.events.queue.events import EndOfNightEvent
from scheduler.graphql_mid.types import NightPlansError
from scheduler.night_monitor.night_monitor import NightMonitor
from scheduler.services.logger_factory import create_logger
from scheduler.engine import SchedulerParameters
from scheduler.engine import EngineRT
from scheduler.shared_queue import plan_response_queue

_logger = create_logger(__name__)

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
        self.params = params
        self.running_event = asyncio.Event()

    def stop_process(self):
        """
        Stop the scheduler process
        """

        _logger.info("Stopping scheduler process...")
        self.running_event.clear()
        # Cancel the running task if needed
        if hasattr(self, 'task'):
            self.task.cancel()

    def start_task(self):
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
        self.running_event.set()

        # Loop through nights until stopped or reached the specified number of nights/date
        night_index = 0
        current_night = self.params.start + TimeDelta(night_index, format='jd')

        while self.running_event.is_set():
            # Check if we have reached the end date or number of nights
            if (self.params.end and current_night >= self.params.end) or \
                 (self.params.num_nights_to_schedule and night_index >= self.params.num_nights_to_schedule):
                _logger.info("End date reached, ending scheduler process.")
                return

            # Initialize the night monitor
            _logger.debug("Initializing night monitor...")
            night_monitor = NightMonitor(current_night, self.params.sites)

            # Initialize Real Time Engine
            _logger.debug("Initializing real-time engine...")
            engine = EngineRT(self.params)

            # Run event loop while still in the same night
            while True:
                # Wait for the next event
                event = await night_monitor.scheduler_queue.get()
                _logger.debug(f"Received scheduler event: {event}")

                # Check if we have reached the end of the night
                if isinstance(event, EndOfNightEvent):
                    _logger.info("Night end event received, ending night scheduling loop.")
                    night_index += 1
                    current_night = self.params.start + TimeDelta(night_index, format='jd')
                    _logger.debug(f"Next night: {current_night}")
                    break

                try:
                    plan = engine.compute_event_plan(event)
                    await plan_response_queue[self.process_id].put(plan)
                except asyncio.CancelledError:
                    _logger.info("Scheduler process was cancelled.")
                except Exception as e:
                    _logger.error(f"Error in scheduler process: {e}")
                    await plan_response_queue[self.process_id].put(NightPlansError(error=str(e)))
                finally:
                    return