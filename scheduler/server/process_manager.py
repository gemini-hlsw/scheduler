# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from datetime import datetime, timezone, timedelta
from astropy.time import Time
from typing import Dict, Optional
from scheduler.core.builder.modes import is_operation
from scheduler.services.logger_factory import create_logger
from scheduler.server.scheduler_process import SchedulerProcess
from scheduler.engine import SchedulerParameters
from scheduler.shared_queue import plan_response_queue

_logger = create_logger(__name__, with_id=False)

__all__ = ["ProcessManager"]

class ProcessManager:
    """
    Manages multiple scheduler processes, allowing for their creation, monitoring, and termination.
    It will keep track of all active scheduler processes and ensure they run as expected.
    Every scheduler process will be identified by a unique process ID provided by the UI subscription
    if running in validation or simulation mode, if running in operation mode it will be "operation".
    """

    def __init__(self):
        """
        Initialize the ProcessManager and the processes' dictionary.
        """
        # Initialize the process manager
        # If operation_mode initialize single scheduler process
        self.active_processes: Dict[str, SchedulerProcess] = {}
        self.operation_process_id: Optional[str] = None
        self._lock = asyncio.Lock()


    async def add_scheduler_process(
            self,
            process_id: str,
            request_params: Optional[SchedulerParameters]
    ):
        """
        Add a new scheduler process to the manager.
        
        Args:
                process_id (str): A unique identifier for the scheduler process.
                request_params (SchedulerParameters): The parameters for the scheduler process.
        """
        # Register a response queue for this process_id so EngineRT can always find it
        if process_id not in plan_response_queue:
            plan_response_queue[process_id] = asyncio.Queue()
        self.active_processes[process_id] = SchedulerProcess(process_id, request_params)
        await self.active_processes[process_id].start_task()

    async def stop_process(self, process_id: str):
        """
        Stop a scheduler process and remove it from the manager.
        
        Args:
                process_id (str): The unique identifier of the scheduler process to stop.
        """
        if process_id in self.active_processes:
            await self.active_processes[process_id].stop_process()
            del self.active_processes[process_id]

    async def set_operation_process(self, process_id: str):
        # We add the process without parameters as those should be setup separately

        params = SchedulerParameters(
            #start=datetime.now(timezone.utc),
            #end=datetime.now(timezone.utc)+timedelta(days=5),
            start=datetime.strptime("2026-01-20 08:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc),
            end=datetime.strptime("2026-01-25 08:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc),
            semester_visibility=False,
            num_nights_to_schedule=1,
            programs_list=["p-cc9", "p-113"]
        )
        await self.add_scheduler_process(process_id, params)
        self.operation_process_id = process_id
        _logger.info(f"Set operation process ID: {process_id}")

    def get_operation_process(self) -> Optional[SchedulerProcess]:
        return self.active_processes[self.operation_process_id]

    async def clear_operation_process(self):
        await self.stop_process(self.operation_process_id)
        self.operation_process_id = None

    async def start(self):
        """
        Start the process manager and monitor all scheduler processes.
        
        If in operation mode, start a single scheduler process with the current time as start.
        Otherwise, listen for new scheduler process requests from the scheduler_process_queue.
        """
        if is_operation:
            await self.set_operation_process("operation")
        else:
            # Start new_process_queue check
            pass
        _logger.info("Process manager started.")
        # Monitor all scheduler processes
        #while True:
        #    for process_id, process in self.active_processes.items():
        #        if not process.is_running:
        #            # Let the UI know that the process has stopped
        #            _logger.warning(f"Scheduler process {process_id} has stopped unexpectedly.")
        #    await asyncio.sleep(60)  # Check every minute

    async def stop(self):
        for process in list(self.active_processes.values()):
            await process.stop_process()
        self.active_processes.clear()
        self.operation_process_id = None
        _logger.info("Process manager stopped.")


process_manager = ProcessManager()