# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from datetime import datetime, timezone
from astropy.time import Time
from typing import Dict
from scheduler.core.builder.modes import is_operation
from scheduler.services.logger_factory import create_logger
from scheduler.server.scheduler_process import SchedulerProcess
from scheduler.engine import SchedulerParameters
from scheduler.shared_queue import scheduler_process_queue

_logger = create_logger(__name__)

__all__ = ["ProcessManager"]

class ProcessManager:
  def __init__(self):
    # Initialize the process manager
    # If operation_mode initialize single scheduler process
    self.processes: Dict[str, SchedulerProcess] = {}

  def add_scheduler_process(self, process_id: str, request_params: SchedulerParameters):
    # Add a new scheduler process
    self.processes[process_id] = SchedulerProcess(process_id, request_params)
    self.processes[process_id].start_task()

  def stop_process(self, process_id: str):
    # Stop the scheduler process
    if process_id in self.processes:
      self.processes[process_id].stop_process()
      del self.processes[process_id]

  async def start(self):
    if is_operation:
      self.add_scheduler_process("operation", SchedulerParameters(
        start=Time(datetime.now(tz=timezone.utc), scale='utc'),
      ))
    else:
      # Start new_process_queue check
      asyncio.create_task(self.new_process_queue())

    # Monitor all scheduler processes
    while True:
      for process_id, process in self.processes.items():
        if not process.is_running:
          # Let the UI know that the process has stopped
          _logger.warning(f"Scheduler process {process_id} has stopped unexpectedly.")
      await asyncio.sleep(60)  # Check every minute

  async def new_process_queue(self):
    while True:
      process_id, request_params = await scheduler_process_queue.get()
      print(f"Process Manager: Adding new scheduler process {process_id}")
      self.add_scheduler_process(process_id, request_params)
