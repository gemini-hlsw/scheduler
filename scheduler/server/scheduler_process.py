# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from scheduler.graphql_mid.types import NewNightPlans, NightPlansError, SNightTimelines, SRunSummary
from scheduler.services.logger_factory import create_logger
from scheduler.engine import Engine, SchedulerParameters
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



    while self.running_event.is_set():
      try:
        engine = Engine(self.params)
        plan_summary, timelines = engine.schedule()
        s_timelines = SNightTimelines.from_computed_timelines(timelines)
        s_plan_summary = SRunSummary.from_computed_run_summary(plan_summary)
        await plan_response_queue[self.process_id].put(NewNightPlans(night_plans=s_timelines, plans_summary=s_plan_summary))
      except asyncio.CancelledError:
        _logger.info("Scheduler process was cancelled.")
      except Exception as e:
        _logger.error(f"Error in scheduler process: {e}")
        await plan_response_queue[self.process_id].put(NightPlansError(error=str(e)))
      finally:
        return
    return